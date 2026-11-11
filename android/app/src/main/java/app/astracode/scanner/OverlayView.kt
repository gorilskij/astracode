package app.astracode.scanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

/** Draws the outline of the code currently in view: yellow while checking, green once confirmed. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private val path = Path()
    private var visible = false
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeWidth = 5 * resources.displayMetrics.density
    }

    /** [corners] is x0, y0, ... in view pixels, or null to clear. */
    fun show(corners: FloatArray?, confirmed: Boolean) {
        visible = corners != null
        if (corners != null) {
            path.rewind()
            path.moveTo(corners[0], corners[1])
            for (i in 1 until corners.size / 2) path.lineTo(corners[2 * i], corners[2 * i + 1])
            path.close()
            paint.color = if (confirmed) 0xFF3DDC84.toInt() else 0xFFFEBC5C.toInt()
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (visible) canvas.drawPath(path, paint)
    }
}
