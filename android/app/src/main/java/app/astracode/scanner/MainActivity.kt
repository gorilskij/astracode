package app.astracode.scanner

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Size
import android.view.HapticFeedbackConstants
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import app.astracode.scanner.decode.Decoder
import app.astracode.scanner.decode.Gray
import app.astracode.scanner.decode.ScanResult
import app.astracode.scanner.decode.Spec
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * Live scanner. Every camera frame goes through the decoder; a number is accepted
 * as soon as two frames in a row agree on it, then scanning pauses and the result
 * is shown until "Scan again".
 */
class MainActivity : ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var hint: TextView
    private lateinit var allow: Button
    private lateinit var panel: View
    private lateinit var numberView: TextView
    private lateinit var detailView: TextView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var luma = ByteArray(0)

    @Volatile private var paused = false
    private var candidate = -1L          // number seen in the previous frame; analysis thread only
    private var result = -1L

    private val askPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showPermissionNeeded()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        hint = findViewById(R.id.hint)
        allow = findViewById(R.id.allow)
        panel = findViewById(R.id.panel)
        numberView = findViewById(R.id.number)
        detailView = findViewById(R.id.detail)

        val hintTop = (hint.layoutParams as android.widget.FrameLayout.LayoutParams).topMargin
        val panelBottom = panel.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            (hint.layoutParams as android.widget.FrameLayout.LayoutParams).topMargin = hintTop + bars.top
            hint.requestLayout()
            panel.updatePadding(bottom = panelBottom + bars.bottom)
            insets
        }

        allow.setOnClickListener { askPermission.launch(Manifest.permission.CAMERA) }
        findViewById<Button>(R.id.copy).setOnClickListener {
            val clipboard = getSystemService(ClipboardManager::class.java)
            clipboard.setPrimaryClip(ClipData.newPlainText(getString(R.string.app_name), Spec.memberId(result % Spec.PAYLOAD_LIMIT)))
            (it as Button).setText(R.string.copied)
        }
        findViewById<Button>(R.id.again).setOnClickListener { resumeScanning() }

        if (savedInstanceState != null && savedInstanceState.getLong(KEY_RESULT, -1) >= 0) {
            showResult(savedInstanceState.getLong(KEY_RESULT), null)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putLong(KEY_RESULT, result)
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    private fun showPermissionNeeded() {
        hint.setText(R.string.hint_no_camera)
        allow.visibility = View.VISIBLE
    }

    private fun startCamera() {
        allow.visibility = View.GONE
        if (result < 0) hint.setText(R.string.hint_scanning)
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            // Preview and analysis both use the sensor's 4:3 output, so they show the same
            // field of view and analysis coordinates map straight onto the preview.
            val fourByThree = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            val previewUseCase = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(fourByThree).build())
                .build()
                .also { it.surfaceProvider = preview.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(fourByThree)
                        .setResolutionStrategy(
                            ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(analysisExecutor, ::analyze)
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUseCase, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        image.use {
            if (paused) return
            val plane = image.planes[0]          // Y: brightness, which is all the decoder needs
            val buffer = plane.buffer
            val needed = plane.rowStride * image.height
            if (luma.size < needed) luma = ByteArray(needed)
            buffer.rewind()
            buffer.get(luma, 0, minOf(buffer.remaining(), needed))
            val gray = Gray.fromLuma(luma, image.width, image.height, plane.rowStride)
            val best = Decoder.scan(gray).maxByOrNull { it.confidence }
            if (paused) return
            val corners = best?.let { toView(it, image.width, image.height, image.imageInfo.rotationDegrees) }
            val confirmed = best != null && best.number == candidate
            candidate = best?.number ?: -1
            if (confirmed) paused = true
            runOnUiThread {
                if (confirmed) showResult(best!!.number, corners) else if (result < 0) overlay.show(corners, false)
            }
        }
    }

    /** Code outline from analysis-image pixels to view pixels (image rotated upright, then centre-cropped to fill). */
    private fun toView(r: ScanResult, w: Int, h: Int, rotation: Int): FloatArray {
        val uprightW = if (rotation % 180 == 0) w else h
        val uprightH = if (rotation % 180 == 0) h else w
        val scale = max(preview.width / uprightW.toFloat(), preview.height / uprightH.toFloat())
        val dx = (preview.width - uprightW * scale) / 2
        val dy = (preview.height - uprightH * scale) / 2
        val out = FloatArray(r.corners.size)
        for (i in 0 until r.corners.size / 2) {
            val x = r.corners[2 * i]
            val y = r.corners[2 * i + 1]
            val (ux, uy) = when (rotation) {
                90 -> h - y to x
                180 -> w - x to h - y
                270 -> y to w - x
                else -> x to y
            }
            out[2 * i] = ux * scale + dx
            out[2 * i + 1] = uy * scale + dy
        }
        return out
    }

    private fun showResult(number: Long, corners: FloatArray?) {
        paused = true
        result = number
        overlay.show(corners, true)
        // Shown as a member ID: the 10-digit payload plus its two mod-97 check digits.
        val payload = number % Spec.PAYLOAD_LIMIT
        numberView.text = Spec.memberId(payload).chunked(4).joinToString(" ")
        detailView.text = getString(R.string.result_detail, number / Spec.PAYLOAD_LIMIT, payload)
        findViewById<Button>(R.id.copy).setText(R.string.copy)
        hint.visibility = View.GONE
        panel.visibility = View.VISIBLE
        val effect = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.VIRTUAL_KEY
        panel.performHapticFeedback(effect)
    }

    private fun resumeScanning() {
        result = -1
        panel.visibility = View.GONE
        hint.visibility = View.VISIBLE
        overlay.show(null, false)
        analysisExecutor.execute { candidate = -1 }
        paused = false
    }

    private companion object {
        const val KEY_RESULT = "result"
    }
}
