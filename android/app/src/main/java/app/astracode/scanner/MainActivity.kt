package app.astracode.scanner

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Patterns
import android.util.Size
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
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
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * Live scanner. Every camera frame goes to the native scanner. When a code is accepted,
 * scanning pauses and the result is shown until "Scan again".
 *
 * A one-ring code is accepted when two frames in a row agree on it: it has only four check
 * stars, and the second frame costs a fraction of a second. A code with more rings is
 * protected by its own long codeword and is accepted at once.
 */
class MainActivity : ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var top: View
    private lateinit var hint: TextView
    private lateinit var status: TextView
    private lateinit var allow: Button
    private lateinit var panel: View
    private lateinit var kindView: TextView
    private lateinit var numberView: TextView
    private lateinit var messageScroll: View
    private lateinit var messageView: TextView
    private lateinit var detailView: TextView
    private lateinit var notesView: TextView
    private lateinit var openButton: Button
    private lateinit var copyButton: Button

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    @Volatile private var paused = false
    /** What the previous frame showed. Analysis thread only. */
    private var pending: Payload? = null

    /** The result on screen: the scanner's packed output and which of its codes, kept so that it survives rotation. */
    private var shownPacked: ByteArray? = null
    private var shownIndex = 0
    /** What Copy copies, and what Open opens. */
    private var copyText = ""
    private var link: String? = null

    private val askPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showPermissionNeeded()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        top = findViewById(R.id.top)
        hint = findViewById(R.id.hint)
        status = findViewById(R.id.status)
        allow = findViewById(R.id.allow)
        panel = findViewById(R.id.panel)
        kindView = findViewById(R.id.kind)
        numberView = findViewById(R.id.number)
        messageScroll = findViewById(R.id.message_scroll)
        messageView = findViewById(R.id.message)
        detailView = findViewById(R.id.detail)
        notesView = findViewById(R.id.notes)
        openButton = findViewById(R.id.open)
        copyButton = findViewById(R.id.copy)

        val topMargin = (top.layoutParams as FrameLayout.LayoutParams).topMargin
        val panelBottom = panel.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            (top.layoutParams as FrameLayout.LayoutParams).topMargin = topMargin + bars.top
            top.requestLayout()
            panel.updatePadding(bottom = panelBottom + bars.bottom)
            insets
        }

        allow.setOnClickListener { askPermission.launch(Manifest.permission.CAMERA) }
        copyButton.setOnClickListener {
            getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText(getString(R.string.app_name), copyText))
            copyButton.setText(R.string.copied)
        }
        openButton.setOnClickListener {
            try {
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(link)))
            } catch (_: ActivityNotFoundException) {
                // Nothing on this phone opens links; the text can still be copied.
            }
        }
        findViewById<Button>(R.id.again).setOnClickListener { resumeScanning() }

        savedInstanceState?.getByteArray(KEY_RESULT)?.let { packed ->
            val index = savedInstanceState.getInt(KEY_INDEX)
            parseCodes(packed).getOrNull(index)?.let { showResult(it, packed, index, null) }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putByteArray(KEY_RESULT, shownPacked)
        outState.putInt(KEY_INDEX, shownIndex)
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
        if (shownPacked == null) hint.setText(R.string.hint_scanning)
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
            // A code with many rings needs its stars to be around 20 pixels tall, so analyse
            // a larger frame than a one-ring code alone would need.
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(fourByThree)
                        .setResolutionStrategy(ResolutionStrategy(ANALYSIS_SIZE, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
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
            val plane = image.planes[0]          // Y: brightness, which is all the scanner needs
            val start = SystemClock.elapsedRealtimeNanos()
            val packed = NativeScanner.scan(plane.buffer, image.width, image.height, plane.rowStride, FIND_SIZE)
            val millis = ((SystemClock.elapsedRealtimeNanos() - start) / 1_000_000).toInt()
            if (paused) return
            val codes = parseCodes(packed)
            val index = if (codes.isEmpty()) -1 else 0
            val code = codes.firstOrNull()
            val corners = code?.let { toView(it.corners, image.width, image.height, image.imageInfo.rotationDegrees) }
            val accepted = code != null && (code.payload is Payload.Message || code.payload == pending)
            pending = code?.payload
            if (accepted) paused = true
            val (width, height) = image.width to image.height
            runOnUiThread {
                status.text = getString(R.string.status, width, height, millis)
                if (accepted) showResult(code!!, packed!!, index, corners) else if (shownPacked == null) overlay.show(corners, false)
            }
        }
    }

    /** A code's outline from frame pixels to view pixels: the frame turned upright, then centre-cropped to fill the preview. */
    private fun toView(corners: FloatArray, w: Int, h: Int, rotation: Int): FloatArray {
        val uprightW = if (rotation % 180 == 0) w else h
        val uprightH = if (rotation % 180 == 0) h else w
        val scale = max(preview.width / uprightW.toFloat(), preview.height / uprightH.toFloat())
        val dx = (preview.width - uprightW * scale) / 2
        val dy = (preview.height - uprightH * scale) / 2
        val out = FloatArray(corners.size)
        for (i in 0 until corners.size / 2) {
            val x = corners[2 * i]
            val y = corners[2 * i + 1]
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

    private fun showResult(code: ScannedCode, packed: ByteArray, index: Int, corners: FloatArray?) {
        paused = true
        shownPacked = packed
        shownIndex = index
        overlay.show(corners, true)

        link = null
        when (val payload = code.payload) {
            is Payload.Number -> {
                // Metadata 0 is a member ID: the ten-digit payload and its two check digits.
                if (payload.metadata == 0) {
                    kindView.setText(R.string.kind_member_id)
                    copyText = payload.memberId
                    numberView.text = payload.memberId.chunked(4).joinToString(" ")
                } else {
                    kindView.text = getString(R.string.kind_number, payload.metadata)
                    copyText = "%010d".format(payload.payload)
                    numberView.text = copyText.chunked(5).joinToString(" ")
                }
                detailView.text = getString(R.string.detail_number, payload.metadata, "%010d".format(payload.payload))
                numberView.visibility = View.VISIBLE
                messageScroll.visibility = View.GONE
            }
            is Payload.Message -> {
                val text = payload.text
                if (text == null) {
                    // Not text: show the bytes.
                    kindView.setText(R.string.kind_data)
                    copyText = payload.bytes.joinToString(" ") { "%02x".format(it) }
                    messageView.typeface = android.graphics.Typeface.MONOSPACE
                } else {
                    link = text.trim().takeIf { isLink(it) }
                    kindView.setText(if (link != null) R.string.kind_link else R.string.kind_text)
                    copyText = text
                    messageView.typeface = android.graphics.Typeface.DEFAULT
                }
                messageView.text = copyText
                detailView.text = resources.getQuantityString(
                    R.plurals.detail_message, payload.bytes.size, payload.bytes.size, code.rings, payload.level)
                numberView.visibility = View.GONE
                messageScroll.visibility = View.VISIBLE
                limitHeight(messageScroll, MESSAGE_MAX_HEIGHT_DP)
            }
        }

        val notes = buildList {
            if (code.mirrored) add(getString(R.string.note_mirrored))
            if (code.bent) add(getString(R.string.note_bent))
            if (code.repaired > 0) {
                val which = if (code.payload is Payload.Number) R.plurals.note_repaired_stars else R.plurals.note_repaired_pairs
                add(resources.getQuantityString(which, code.repaired, code.repaired))
            }
        }
        notesView.text = notes.joinToString(" · ")
        notesView.visibility = if (notes.isEmpty()) View.GONE else View.VISIBLE

        openButton.visibility = if (link != null) View.VISIBLE else View.GONE
        copyButton.setText(R.string.copy)
        top.visibility = View.GONE
        panel.visibility = View.VISIBLE
        val effect = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.VIRTUAL_KEY
        panel.performHapticFeedback(effect)
    }

    private fun isLink(text: String) =
        (text.startsWith("http://") || text.startsWith("https://")) && Patterns.WEB_URL.matcher(text).matches()

    /** Let a view be as tall as its content, up to [maxDp]; beyond that it scrolls. */
    private fun limitHeight(view: View, maxDp: Int) {
        view.layoutParams = view.layoutParams.apply { height = ViewGroup.LayoutParams.WRAP_CONTENT }
        view.post {
            val limit = (maxDp * resources.displayMetrics.density).toInt()
            if (view.height > limit) view.layoutParams = view.layoutParams.apply { height = limit }
        }
    }

    private fun resumeScanning() {
        shownPacked = null
        panel.visibility = View.GONE
        top.visibility = View.VISIBLE
        overlay.show(null, false)
        analysisExecutor.execute { pending = null }
        paused = false
    }

    private companion object {
        const val KEY_RESULT = "result"
        const val KEY_INDEX = "index"

        /** The frame size asked of the camera for analysis. */
        val ANALYSIS_SIZE = Size(1920, 1440)

        /** Look for stars at full resolution; see [NativeScanner.scan]. */
        const val FIND_SIZE = 0

        const val MESSAGE_MAX_HEIGHT_DP = 180
    }
}
