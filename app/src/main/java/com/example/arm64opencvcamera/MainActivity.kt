package com.example.arm64opencvcamera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.*
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import kotlin.math.sqrt
import androidx.core.graphics.createBitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.scale
import androidx.core.graphics.get

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private lateinit var textViewStatus: TextView
    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button
    private lateinit var buttonSaveSample: Button
    private lateinit var buttonIdentify: Button
    private lateinit var checkBoxProcessing: CheckBox
    private lateinit var inputName: EditText
    private lateinit var imageView: ImageView
    private lateinit var openCvCameraView: CameraBridgeViewBase
    private lateinit var textViewUserName: TextView
    private lateinit var eyeClassifier: CascadeClassifier
    private lateinit var inputMat: Mat
    private lateinit var grayMat: Mat
    private var isPreviewRunning = false
    private var lastIrisMat: Mat? = null
    private var lastIrisBitmap: Bitmap? = null
    private lateinit var tflite: Interpreter

    companion object {
        private const val DETECT_EYES_SCALE_FACTOR = 1.1
        /* Меньшее значение (ближе к 1.0) означает более тщательное сканирование, но медленнее.
        Алгоритм будет искать глаза разных размеров, но это будет занимать больше времени.
        Большее значение (например, 1.2, 1.3) означает более грубое сканирование, но быстрее. */
        private const val DETECT_EYES_MIN_NEIGHBORS = 3
        /* Меньшее значение (например, 1 или 2) означает, что алгоритм будет более чувствительным и
        будет считать больше объектов глазами, даже если это ложные срабатывания.
        Большее значение (например, 4, 5, 6) означает, что алгоритм будет более строгим и будет
        считать меньше объектов глазами, но при этом уменьшается количество ложных срабатываний. */
        private const val IN_RANGE_MIN = 0.0
        /* Устанавливает минимальное значение интенсивности пикселей, которые будут считаться частью
        радужной оболочки. Оптимальное значение: 0 - это стандартное значение для чёрного цвета. */
        private const val IN_RANGE_MAX = 100.0
        /* Устанавливает максимальное значение интенсивности пикселей, которые будут считаться частью
        радужной оболочки. Оптимальное значение: 100 - это значение для относительно тёмных пикселей.
        Для светлых пикселей необходимо увеличить это значение до 150-200. */
        private const val RESIZE_SIZE = 96
        /* Значение: 96 выбрано, потому что именно такой размер ожидает модель. */
        private const val SIMILARITY_THRESHOLD = 0.85f
        /* Если косинусная мера сходства между двумя изображениями больше или равно SIMILARITY_THRESHOLD,
         то изображения считаются одинаковыми.
         •Слишком маленькое значение: Будет много ложных срабатываний.
         •Слишком большое значение: Программа может не распознать человека, даже если это он,
         из-за небольших отличий в изображении. */
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        textViewStatus = findViewById(R.id.textViewStatus)
        buttonStartPreview = findViewById(R.id.buttonStartPreview)
        buttonStopPreview = findViewById(R.id.buttonStopPreview)
        buttonSaveSample = findViewById(R.id.buttonSaveSample)
        buttonIdentify = findViewById(R.id.buttonIdentify)
        checkBoxProcessing = findViewById(R.id.checkboxEnableProcessing)
        inputName = findViewById(R.id.editTextName)
        imageView = findViewById(R.id.imageView)
        openCvCameraView = findViewById(R.id.cameraView)
        textViewUserName = findViewById(R.id.textViewUserName)

        if (!OpenCVLoader.initLocal()) {
            textViewStatus.text = getString(R.string.opencv_initialization_error)
            return
        }

        // Initialize TFLite
        try {
            assets.open("mobilenetv2_embedding_96x96.tflite").use { inputStream ->
                val tfliteModelBytes = inputStream.readBytes()
                val tfliteModelBuffer = ByteBuffer.allocateDirect(tfliteModelBytes.size)
                tfliteModelBuffer.order(ByteOrder.nativeOrder())
                tfliteModelBuffer.put(tfliteModelBytes)
                tfliteModelBuffer.rewind()
                tflite = Interpreter(tfliteModelBuffer)
            }
        } catch (e: Exception) {
            textViewStatus.text = getString(R.string.error_loading_TFLite_model, e.message)
            return
        }

        loadEyeCascade()

        openCvCameraView.visibility = CameraBridgeViewBase.VISIBLE
        openCvCameraView.setCameraPermissionGranted()
        openCvCameraView.setCvCameraViewListener(this)

        checkBoxProcessing.setOnCheckedChangeListener { _, _ -> updateControls() }

        buttonStartPreview.setOnClickListener {
            openCvCameraView.enableView()
            isPreviewRunning = true
            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            openCvCameraView.disableView()
            isPreviewRunning = false
            updateControls()
        }

        buttonSaveSample.setOnClickListener {
            saveIrisSample()
        }

        buttonIdentify.setOnClickListener {
            identifyIris()
        }

        updateControls()
    }

    private fun updateControls() {
        buttonStartPreview.isEnabled = !isPreviewRunning
        buttonStopPreview.isEnabled = isPreviewRunning
        buttonSaveSample.isEnabled = isPreviewRunning && checkBoxProcessing.isChecked
        buttonIdentify.isEnabled = isPreviewRunning && checkBoxProcessing.isChecked
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        inputMat = Mat(height, width, CvType.CV_8UC4)
        grayMat = Mat(height, width, CvType.CV_8UC1)
    }

    override fun onCameraViewStopped() {
        inputMat.release()
        grayMat.release()
        lastIrisMat?.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        inputFrame!!.rgba().copyTo(inputMat)
        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.equalizeHist(grayMat, grayMat)

        if (checkBoxProcessing.isChecked) {
            val eyes = MatOfRect()
            eyeClassifier.detectMultiScale(grayMat, eyes, DETECT_EYES_SCALE_FACTOR, DETECT_EYES_MIN_NEIGHBORS)
            for (eyeRect in eyes.toArray()) {
                if (eyeRect.width > grayMat.width() / 10 && eyeRect.height > grayMat.height() / 10) {
                    val eyeROI = grayMat.submat(eyeRect)
                    val irisMask = Mat()
                    val contours = mutableListOf<MatOfPoint>()
                    try {
                        Core.inRange(eyeROI, Scalar(IN_RANGE_MIN), Scalar(IN_RANGE_MAX), irisMask)

                        Imgproc.findContours(
                            irisMask,
                            contours,
                            Mat(),
                            Imgproc.RETR_EXTERNAL,
                            Imgproc.CHAIN_APPROX_SIMPLE
                        )

                        var largestArea = 0.0
                        var largestContour: MatOfPoint? = null
                        for (contour in contours) {
                            val area = Imgproc.contourArea(contour)
                            if (area > largestArea) {
                                largestArea = area
                                largestContour = contour
                            }
                        }
                        largestContour?.let {
                            val m = Imgproc.moments(it)
                            val cx = (m.m10 / m.m00).toInt()
                            val cy = (m.m01 / m.m00).toInt()
                            val radius = sqrt(largestArea / Math.PI).toInt()
                            val center = Point((eyeRect.x + cx).toDouble(), (eyeRect.y + cy).toDouble())
                            Imgproc.circle(inputMat, center, radius, Scalar(255.0, 0.0, 0.0), 2)
                            lastIrisMat = eyeROI.clone()
                            // Convert Mat to Bitmap for TFLite
                            val bitmap = createBitmap(eyeROI.cols(), eyeROI.rows())
                            Utils.matToBitmap(eyeROI, bitmap)
                            lastIrisBitmap = bitmap
                        }
                        Imgproc.rectangle(inputMat, eyeRect, Scalar(0.0, 255.0, 0.0), 2)
                    } finally {
                        //Освобождаем память для объектов из списка
                        contours.forEach { it.release() }
                        irisMask.release()
                        eyeROI.release()
                    }
                }
            }
        }

        val bitmap = createBitmap(inputMat.cols(), inputMat.rows())
        Utils.matToBitmap(inputMat, bitmap)
        runOnUiThread { imageView.setImageBitmap(bitmap) }

        return inputMat
    }

    private fun loadEyeCascade() {
        try {
            assets.open("haarcascade_eye.xml").use { inputStream ->
                val cascadeDir = getDir("cascade", MODE_PRIVATE)
                val cascadeFile = File(cascadeDir, "haarcascade_eye.xml")
                FileOutputStream(cascadeFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
                eyeClassifier = CascadeClassifier(cascadeFile.absolutePath)
                cascadeDir.deleteRecursively()
            }
        } catch (e: Exception) {
            textViewStatus.text = getString(R.string.error_loading_cascade, e.message)
        }
    }

    private fun getIrisEmbedding(bitmap: Bitmap): FloatArray {
        val resized = bitmap.scale(RESIZE_SIZE, RESIZE_SIZE)
        val input = ByteBuffer.allocateDirect(96 * 96 * 3 * 4)
        input.order(ByteOrder.nativeOrder())
        for (y in 0 until 96) {
            for (x in 0 until 96) {
                val pixel = resized[x, y]
                input.putFloat(Color.red(pixel) / 255.0f)
                input.putFloat(Color.green(pixel) / 255.0f)
                input.putFloat(Color.blue(pixel) / 255.0f)
            }
        }
        val output = Array(1) { FloatArray(1280) } // MobileNetV2 global average pooling output
        try {
            tflite.run(input.rewind(), output)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error during TFLite inference: ${e.message}")
            // You can add a default return. Example return an empty array:
            return floatArrayOf()
        }
        return output[0]
    }
    // расчет косинусной меры сходства
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        return dot / (sqrt(normA.toDouble()) * sqrt(normB.toDouble())).toFloat()
    }

    private fun saveIrisSample() {
        val name = inputName.text.toString().trim()
        if (name.isEmpty() || lastIrisBitmap == null) {
            Toast.makeText(this, "Enter a name and ensure iris is detected", Toast.LENGTH_SHORT)
                .show()
            return
        }

        try {
            val emb = getIrisEmbedding(lastIrisBitmap!!)
            if (emb.isEmpty()){
                Toast.makeText(this, "Error save data", Toast.LENGTH_SHORT).show()
                return
            }
            File(filesDir, "$name.emb").printWriter().use { out ->
                emb.forEach { out.println(it) }
            }
            Toast.makeText(this, "Embedding saved for $name", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error saving embedding: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun identifyIris() {
        if (lastIrisBitmap == null) {
            Toast.makeText(this, "No valid iris to identify", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val inputEmb = getIrisEmbedding(lastIrisBitmap!!)
            if (inputEmb.isEmpty()){
                Toast.makeText(this, "Error get data", Toast.LENGTH_SHORT).show()
                return
            }
            val files = filesDir.listFiles { it -> it.extension == "emb" } ?: return
            var bestMatch = "Unknown"
            var bestScore = -1f

            for (file in files) {
                val saved = file.readLines().map { it.toFloat() }.toFloatArray()
                val sim = cosineSimilarity(inputEmb, saved)
                if (sim > bestScore) {
                    bestScore = sim
                    bestMatch = file.nameWithoutExtension
                }
            }

            if (bestScore < SIMILARITY_THRESHOLD) {
                bestMatch = "Unknown"
            }

            Toast.makeText(this, "Identified as $bestMatch", Toast.LENGTH_SHORT).show()
            runOnUiThread {
                textViewStatus.text = getString(R.string.identification_result, bestMatch)
                textViewUserName.text = bestMatch
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error identifying iris: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}