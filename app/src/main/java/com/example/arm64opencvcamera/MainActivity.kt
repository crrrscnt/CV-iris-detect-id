package com.example.arm64opencvcamera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
//import android.widget.CheckBox
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import androidx.core.view.isVisible

import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private lateinit var textViewStatus: TextView
    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button
    private lateinit var buttonSwitchCamera: Button
    private lateinit var buttonIdentify: Button
//    private lateinit var checkBoxProcessing: CheckBox
    private lateinit var imageView: ImageView // For camera preview
    private lateinit var openCvCameraView: CameraBridgeViewBase
    private lateinit var textViewUserName: TextView
    private lateinit var eyeClassifier: CascadeClassifier
    private lateinit var inputMat: Mat // For camera frames
    private lateinit var grayMat: Mat // For camera frames
    private var isPreviewRunning = false
    private var lastIrisMat: Mat? = null // Iris Mat from camera
    private var lastIrisBitmap: Bitmap? = null // Iris Bitmap from camera for identification
    private lateinit var tflite: Interpreter

    private var currentCameraId = CameraBridgeViewBase.CAMERA_ID_BACK
    private var liveen = false
    // Удалены UI элементы для обработки фото из галереи
    // private lateinit var buttonSelectPhoto: Button
    // private lateinit var layoutPreviewContainer: LinearLayout
    // private lateinit var imageViewPreview: ImageView
    // private lateinit var editTextUserNamePreview: EditText
    // private lateinit var buttonSaveExamplePreview: Button
    // private var detectedIrisBitmapFromGallery: Bitmap? = null


//    private var userdb: UserDatabase? = null

    companion object {
        private const val DETECT_EYES_SCALE_FACTOR = 1.1
        private const val DETECT_EYES_MIN_NEIGHBORS = 3
        private const val IN_RANGE_MIN = 0.0
        private const val IN_RANGE_MAX = 100.0
        private const val RESIZE_SIZE = 96
        private const val SIMILARITY_THRESHOLD = 0.7f
        private const val REQUEST_CODE_PERMISSIONS = 101

        private const val IDENTIFICATION_INTERVAL_MS = 3000
        private val REQUIRED_PERMISSIONS = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        private const val IRIS_SAMPLES_FOLDER_NAME = "IrisRecognition"
    }

    // ActivityResultLauncher для выбора изображения из галереи УДАЛЕН
    // private val pickImageLauncher = ...

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        textViewStatus = findViewById(R.id.textViewStatus)
        buttonStartPreview = findViewById(R.id.buttonStartPreview)
        buttonStopPreview = findViewById(R.id.buttonStopPreview)
        buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera)
        buttonIdentify = findViewById(R.id.buttonIdentify)
//        checkBoxProcessing = findViewById(R.id.checkboxEnableProcessing)
        imageView = findViewById(R.id.imageView)
        openCvCameraView = findViewById(R.id.cameraView)
        textViewUserName = findViewById(R.id.textViewUserName)

        // Инициализация UI элементов для галереи УДАЛЕНА
        // buttonSelectPhoto = findViewById(R.id.buttonSelectPhoto)
        // layoutPreviewContainer = findViewById(R.id.layoutPreviewContainer)
        // imageViewPreview = findViewById(R.id.imageViewPreview)
        // editTextUserNamePreview = findViewById(R.id.editTextUserNamePreview)
        // buttonSaveExamplePreview = findViewById(R.id.buttonSaveExamplePreview)
        // layoutPreviewContainer.visibility = View.GONE


        if (!allPermissionsGranted()) {
            requestPermissions(REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        } else {
            initializeOpenCVAndTF()
            // Сканируем папку с образцами, если разрешения уже есть
            scanAndProcessIrisSamplesDirectory()
        }


        openCvCameraView.visibility = CameraBridgeViewBase.VISIBLE
        openCvCameraView.setCameraPermissionGranted() // Предполагаем, что разрешение уже есть или будет запрошено
        openCvCameraView.setCameraIndex(currentCameraId)
        openCvCameraView.setCvCameraViewListener(this)

//        checkBoxProcessing.setOnCheckedChangeListener { _, _ -> updateControls() }

        buttonStartPreview.setOnClickListener {
            openCvCameraView.setCameraIndex(currentCameraId)
            openCvCameraView.enableView()
            isPreviewRunning = true
            // layoutPreviewContainer.visibility = View.GONE // Удалено
            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            openCvCameraView.disableView()
            isPreviewRunning = false
            updateControls()
        }

        buttonSwitchCamera.setOnClickListener {
            switchCamera()
        }

//        buttonIdentify.setOnClickListener {
//            identifyIris()
//        }
        buttonIdentify.setOnClickListener {
            if (!isPreviewRunning) {
                openCvCameraView.setCameraIndex(currentCameraId)
                openCvCameraView.enableView()
                isPreviewRunning = true
            }

            if (!liveen) {
                liveen = true
                Toast.makeText(this, "Live identification started", Toast.LENGTH_SHORT).show()

                val handler = android.os.Handler(mainLooper)
                handler.post(object : Runnable {
                    override fun run() {
                        if (liveen && isPreviewRunning) {
                            identifyIris()
                            handler.postDelayed(this, IDENTIFICATION_INTERVAL_MS.toLong())
                        }
                    }
                })
            }
        }

        // Listener для кнопки "Select Photo" УДАЛЕН
        // buttonSelectPhoto.setOnClickListener { ... }

        // Listener для кнопки "Save Example" УДАЛЕН
        // buttonSaveExamplePreview.setOnClickListener { ... }
//        loadUserDatabase();
        updateControls()
    }



    private fun initializeOpenCVAndTF() {
        if (!OpenCVLoader.initLocal()) {
            textViewStatus.text = getString(R.string.opencv_initialization_error)
            Log.e("MainActivity", "OpenCV initialization failed")
            return
        }
        Log.i("MainActivity", "OpenCV initialized successfully")

        try {
            assets.open("mobilenetv2_embedding_96x96.tflite").use { inputStream ->
                val tfliteModelBytes = inputStream.readBytes()
                val tfliteModelBuffer = ByteBuffer.allocateDirect(tfliteModelBytes.size)
                tfliteModelBuffer.order(ByteOrder.nativeOrder())
                tfliteModelBuffer.put(tfliteModelBytes)
                tfliteModelBuffer.rewind()
                tflite = Interpreter(tfliteModelBuffer)
                Log.i("MainActivity", "TFLite model loaded successfully")
            }
        } catch (e: Exception) {
            textViewStatus.text = getString(R.string.error_loading_TFLite_model, e.message)
            Log.e("MainActivity", "Error loading TFLite model", e)
            return
        }
        loadEyeCascade()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initializeOpenCVAndTF()
                scanAndProcessIrisSamplesDirectory() // Сканируем после получения разрешений
                if (isPreviewRunning) { // Если предпросмотр должен был быть запущен
                    openCvCameraView.setCameraIndex(currentCameraId)
                    openCvCameraView.enableView()
                }
            } else {
                Toast.makeText(this, getString(R.string.permissions_not_granted), Toast.LENGTH_SHORT).show()
                // finish() // Or handle gracefully
            }
        }
    }

    private fun switchCamera() {
        openCvCameraView.disableView()
        currentCameraId = if (currentCameraId == CameraBridgeViewBase.CAMERA_ID_BACK) {
            CameraBridgeViewBase.CAMERA_ID_FRONT
        } else {
            CameraBridgeViewBase.CAMERA_ID_BACK
        }
        openCvCameraView.setCameraIndex(currentCameraId)
        if (isPreviewRunning) {
            openCvCameraView.enableView()
        }
    }

    private fun updateControls() {
        buttonStartPreview.isEnabled = !isPreviewRunning
        buttonStopPreview.isEnabled = isPreviewRunning
        buttonSwitchCamera.isEnabled = true
        buttonIdentify.isEnabled = isPreviewRunning // && checkBoxProcessing.isChecked
        // buttonSelectPhoto.isEnabled = true // Удалено
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        inputMat = Mat(height, width, CvType.CV_8UC4)
        grayMat = Mat(height, width, CvType.CV_8UC1)
        Log.i("MainActivity", "Camera view started")
    }

    override fun onCameraViewStopped() {
        inputMat.release()
        grayMat.release()
        lastIrisMat?.release()
        lastIrisMat = null
        Log.i("MainActivity", "Camera view stopped")
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        if (inputFrame == null) return Mat()

        inputFrame.rgba().copyTo(inputMat)

        if (currentCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT) {
            Core.flip(inputMat, inputMat, 1)
        }

        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.equalizeHist(grayMat, grayMat)

        var detectedIrisInFrame = false
        if (liveen) { // checkBoxProcessing.isChecked
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
                            irisMask, contours, Mat(),
                            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
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
                            if (m.m00 != 0.0) {
                                val cx = (m.m10 / m.m00).toInt()
                                val cy = (m.m01 / m.m00).toInt()
                                val radius = sqrt(largestArea / Math.PI).toInt()
                                val center = Point((eyeRect.x + cx).toDouble(), (eyeRect.y + cy).toDouble())
                                Imgproc.circle(inputMat, center, radius, Scalar(255.0, 0.0, 0.0), 2)

                                lastIrisMat?.release()
                                lastIrisMat = eyeROI.clone()
                                val bitmap = createBitmap(eyeROI.cols(), eyeROI.rows(), Bitmap.Config.ARGB_8888)
                                Utils.matToBitmap(eyeROI, bitmap)
                                lastIrisBitmap = bitmap
                                detectedIrisInFrame = true
                            }
                        }
                        Imgproc.rectangle(inputMat, eyeRect, Scalar(0.0, 255.0, 0.0), 2)
                    } finally {
                        contours.forEach { c -> c.release() }
                        irisMask.release()
                        eyeROI.release()
                    }
                    if (detectedIrisInFrame) break
                }
            }
            eyes.release()
        }
        if (!detectedIrisInFrame) {
            lastIrisBitmap = null
        }

        val displayBitmap = createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, displayBitmap)
        runOnUiThread { imageView.setImageBitmap(displayBitmap) }

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
                if (eyeClassifier.empty()) {
                    Log.e("MainActivity", "Failed to load cascade classifier: empty classifier")
                    textViewStatus.text = getString(R.string.error_loading_cascade, "empty classifier")
                } else {
                    Log.i("MainActivity", "Loaded cascade classifier successfully from ${cascadeFile.absolutePath}")
                }
                cascadeFile.delete()
                cascadeDir.delete()
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading cascade", e)
            textViewStatus.text = getString(R.string.error_loading_cascade, e.message)
        }
    }

    // --- Методы для обработки фото из галереи УДАЛЕНЫ ---
    // private fun processSelectedImageUri(uri: Uri) { ... }
    // private fun detectAndProcessIrisFromBitmap(sourceBitmap: Bitmap) { ... }
    // private fun saveIrisEmbeddingFromGallery() { ... }


    // --- Новые методы для сканирования папки и сохранения образцов ---
    private fun scanAndProcessIrisSamplesDirectory() {
        if (!allPermissionsGranted()) {
            Log.w("MainActivity", "Storage permissions not granted. Cannot scan directory.")
            // Toast.makeText(this, "Storage permission needed to load samples.", Toast.LENGTH_LONG).show() // Убрано, т.к. вызывается после проверки
            return
        }
        if (!::eyeClassifier.isInitialized || eyeClassifier.empty()) {
            Log.e("MainActivity", "Eye cascade classifier not loaded. Cannot process images.")
            Toast.makeText(this, "Error: Eye classifier not ready.", Toast.LENGTH_SHORT).show()
            return
        }


        val irisFolderPath = Environment.getExternalStorageDirectory().path + File.separator + IRIS_SAMPLES_FOLDER_NAME
        val irisFolder = File(irisFolderPath)

        if (!irisFolder.exists()) {
            Log.i("MainActivity", "$IRIS_SAMPLES_FOLDER_NAME folder does not exist: $irisFolderPath")
            Toast.makeText(this, "Folder 'Internal storage/$IRIS_SAMPLES_FOLDER_NAME' not found.", Toast.LENGTH_LONG).show()
            return
        }

        if (!irisFolder.isDirectory) {
            Log.e("MainActivity", "$IRIS_SAMPLES_FOLDER_NAME is not a directory: $irisFolderPath")
            return
        }

        val imageFiles = irisFolder.listFiles { _, name ->
            name.endsWith(".jpg", ignoreCase = true) ||
                    name.endsWith(".jpeg", ignoreCase = true) ||
                    name.endsWith(".png", ignoreCase = true)
        }

        if (imageFiles == null || imageFiles.isEmpty()) {
            Log.i("MainActivity", "No image files found in $irisFolderPath")
            // Toast.makeText(this, "No new images found in $IRIS_SAMPLES_FOLDER_NAME.", Toast.LENGTH_SHORT).show()
            return
        }

        var samplesProcessed = 0
        var errorsOccurred = 0
        runOnUiThread { textViewStatus.text = "Processing images from $IRIS_SAMPLES_FOLDER_NAME..." }

        // Обработка в фоновом потоке, чтобы не блокировать UI
        Thread {
            for (imageFile in imageFiles) {
                try {
                    val imageNameWithoutExtension = imageFile.nameWithoutExtension
                    val embFile = File(filesDir, "$imageNameWithoutExtension.emb")

                    Log.i("MainActivity", "Processing image: ${imageFile.name}")
                    val bitmapOptions = BitmapFactory.Options().apply {
                        inPreferredConfig = Bitmap.Config.ARGB_8888
                    }
                    val sourceBitmap = BitmapFactory.decodeFile(imageFile.absolutePath, bitmapOptions)

                    if (sourceBitmap == null) {
                        Log.e("MainActivity", "Failed to decode image: ${imageFile.name}")
                        errorsOccurred++
                        continue
                    }

                    val detectedIrisBitmap = extractIrisFromImage(sourceBitmap)

                    if (detectedIrisBitmap != null) {
                        val colorIrisBitmapForEmbedding = convertGrayscaleToColorBitmap(detectedIrisBitmap)
                        if (colorIrisBitmapForEmbedding != null) {
                            val finalEmb = getIrisEmbedding(colorIrisBitmapForEmbedding)
                            if (finalEmb.isNotEmpty()) {
                                embFile.printWriter().use { out ->
                                    finalEmb.forEach { out.println(it) }
                                }
                                Log.i("MainActivity", "Saved embedding for ${imageFile.name} as ${embFile.name}")
                                samplesProcessed++
                            } else {
                                Log.e("MainActivity", "Failed to generate embedding for ${imageFile.name}")
                                errorsOccurred++
                            }
                            colorIrisBitmapForEmbedding.recycle()
                        } else {
                            Log.e("MainActivity", "Failed to convert detected iris to color for ${imageFile.name}")
                            errorsOccurred++
                        }
                        detectedIrisBitmap.recycle()
                    } else {
                        Log.w("MainActivity", "No iris detected in ${imageFile.name}")
                    }
                    sourceBitmap.recycle()

                } catch (e: Exception) {
                    Log.e("MainActivity", "Error processing file ${imageFile.name}", e)
                    errorsOccurred++
                }
            }
            runOnUiThread {
                var statusMessage = ""
                if (samplesProcessed > 0) {
                    statusMessage += "$samplesProcessed iris samples processed from $IRIS_SAMPLES_FOLDER_NAME. "
                } else if (imageFiles.isNotEmpty() && errorsOccurred == 0) {
                    statusMessage += "No new iris samples processed or found in $IRIS_SAMPLES_FOLDER_NAME. "
                }
                if (errorsOccurred > 0) {
                    statusMessage += "$errorsOccurred errors during processing."
                }
                if (statusMessage.isEmpty()){
                    statusMessage = "Finished scanning $IRIS_SAMPLES_FOLDER_NAME. No images found or processed."
                }
                textViewStatus.text = statusMessage.trim()
                Toast.makeText(this, statusMessage.trim(), Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    private fun extractIrisFromImage(sourceBitmap: Bitmap): Bitmap? {
        val processingMat = Mat()
        Utils.bitmapToMat(sourceBitmap, processingMat)

        val localGrayMat = Mat()
        if (processingMat.channels() >= 3) { // Проверяем, есть ли хотя бы 3 канала (RGB или RGBA)
            Imgproc.cvtColor(processingMat, localGrayMat, Imgproc.COLOR_RGB2GRAY)
        } else if (processingMat.channels() == 1) { // Если уже монохромное
            processingMat.copyTo(localGrayMat)
        } else { // Неожиданное количество каналов
            Log.e("ExtractIris", "Unsupported number of channels in source image: ${processingMat.channels()}")
            processingMat.release()
            return null
        }
        Imgproc.equalizeHist(localGrayMat, localGrayMat)

        val eyes = MatOfRect()
        // Убедимся, что eyeClassifier инициализирован
        if (!::eyeClassifier.isInitialized || eyeClassifier.empty()) {
            Log.e("ExtractIris", "Eye cascade classifier not loaded.")
            processingMat.release()
            localGrayMat.release()
            eyes.release()
            return null
        }
        eyeClassifier.detectMultiScale(localGrayMat, eyes, DETECT_EYES_SCALE_FACTOR, DETECT_EYES_MIN_NEIGHBORS)

        var extractedIrisBitmap: Bitmap? = null

        for (eyeRect in eyes.toArray()) {
            if (eyeRect.width > localGrayMat.width() / 10 && eyeRect.height > localGrayMat.height() / 10) {
                val eyeROI = localGrayMat.submat(eyeRect)
                val irisMask = Mat()
                val contours = mutableListOf<MatOfPoint>()
                try {
                    Core.inRange(eyeROI, Scalar(IN_RANGE_MIN), Scalar(IN_RANGE_MAX), irisMask)
                    Imgproc.findContours(
                        irisMask, contours, Mat(),
                        Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
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
                        if (m.m00 != 0.0) {
                            val irisForEmbeddingBitmap = createBitmap(eyeROI.cols(), eyeROI.rows(), Bitmap.Config.ARGB_8888)
                            Utils.matToBitmap(eyeROI, irisForEmbeddingBitmap)
                            extractedIrisBitmap = irisForEmbeddingBitmap
                            // Не рисуем на изображении, т.к. это автоматическая обработка
                        }
                    }
                } finally {
                    contours.forEach { c -> c.release() }
                    irisMask.release()
                    eyeROI.release()
                }
                if (extractedIrisBitmap != null) break
            }
        }
        eyes.release()
        localGrayMat.release()
        processingMat.release()
        return extractedIrisBitmap
    }


    // --- Общая логика эмбеддинга и идентификации ---

    private fun convertGrayscaleToColorBitmap(grayBitmap: Bitmap): Bitmap? {
        if (grayBitmap.config == Bitmap.Config.ARGB_8888) {
            // Проверяем, действительно ли это оттенки серого в формате ARGB_8888 (R=G=B)
            // Это ожидаемый формат от Utils.matToBitmap для монохромного Mat
            return grayBitmap
        }
        if (grayBitmap.config == Bitmap.Config.RGB_565) { // Также может быть цветным
            return grayBitmap.copy(Bitmap.Config.ARGB_8888, true) // Конвертируем в ARGB_8888
        }


        val width = grayBitmap.width
        val height = grayBitmap.height
        // Создаем ARGB_8888, т.к. TFLite модель ожидает 3 канала
        val colorBitmap = createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        grayBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            // Предполагаем, что grayBitmap был ALPHA_8 или аналогом, где значение серого в альфа-канале
            // или что это ARGB_8888, где R=G=B. Для безопасности берем красный канал.
            val gray = Color.red(pixels[i])
            pixels[i] = Color.rgb(gray, gray, gray)
        }
        colorBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return colorBitmap
    }


    private fun getIrisEmbedding(bitmap: Bitmap): FloatArray {
        val inputBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            bitmap
        }

        val resized = inputBitmap.scale(RESIZE_SIZE, RESIZE_SIZE)
        val inputBuffer = ByteBuffer.allocateDirect(RESIZE_SIZE * RESIZE_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.rewind()

        val intValues = IntArray(RESIZE_SIZE * RESIZE_SIZE)
        resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)

        for (pixelValue in intValues) {
            inputBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f) // Red
            inputBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)  // Green
            inputBuffer.putFloat((pixelValue and 0xFF) / 255.0f)          // Blue
        }
        resized.recycle() // Освобождаем измененный bitmap

        val output = Array(1) { FloatArray(1280) }
        try {
            if (!::tflite.isInitialized) {
                Log.e("getIrisEmbedding", "TFLite interpreter not initialized.")
                return floatArrayOf()
            }
            tflite.run(inputBuffer, output)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error during TFLite inference: ${e.message}", e)
            runOnUiThread { Toast.makeText(this, "TFLite inference error: ${e.localizedMessage}", Toast.LENGTH_LONG).show()}
            return floatArrayOf()
        }
        return output[0]
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) {
            Log.w("CosineSimilarity", "Input arrays are invalid for cosine similarity. A size: ${a.size}, B size: ${b.size}")
            return 0f
        }
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val normProduct = sqrt(normA.toDouble()) * sqrt(normB.toDouble())
        return if (normProduct == 0.0) 0f else (dot / normProduct).toFloat()
    }

    private fun identifyIris() {
        val irisToIdentify = lastIrisBitmap

        if (irisToIdentify == null) {
            Toast.makeText(this, getString(R.string.no_iris_to_identify), Toast.LENGTH_SHORT).show()
            return
        }

        val colorIrisToIdentify = convertGrayscaleToColorBitmap(irisToIdentify)
        if (colorIrisToIdentify == null) {
            Toast.makeText(this, "Error: Could not convert iris to color for identification.", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val inputEmb = getIrisEmbedding(colorIrisToIdentify)
            // colorIrisToIdentify.recycle() // Освобождаем после получения эмбеддинга, если он копировался

            if (inputEmb.isEmpty()) {
                Toast.makeText(this, getString(R.string.error_embedding_generation_for_identification), Toast.LENGTH_SHORT).show()
                return
            }
            val embDir = filesDir
            val files = embDir.listFiles { _, name -> name.endsWith(".emb") }
            if (files == null || files.isEmpty()) {
                Toast.makeText(this, getString(R.string.no_saved_samples), Toast.LENGTH_SHORT).show()
                return
            }

            var bestMatch = "Unknown"
            var bestScore = -1f

            for (file in files) {
                try {
                    val savedEmb = file.readLines().mapNotNull { it.toFloatOrNull() }.toFloatArray()
                    if (savedEmb.isEmpty() || savedEmb.size != inputEmb.size) {
                        Log.w("IdentifyIris", "Skipping invalid or mismatched embedding file: ${file.name}")
                        continue
                    }
                    val sim = cosineSimilarity(inputEmb, savedEmb)
                    Log.i("IdentifyIris", "Comparing with ${file.nameWithoutExtension}, Similarity: $sim")
                    if (sim > bestScore) {
                        bestScore = sim
                        bestMatch = file.nameWithoutExtension
                    }
                } catch (e: Exception) {
                    Log.e("IdentifyIris", "Error reading or processing file ${file.name}", e)
                }
            }

            val identifiedUserName = if (bestScore >= SIMILARITY_THRESHOLD) bestMatch else "Unknown"
            val resultText = getString(R.string.identification_result_detailed, identifiedUserName, String.format(
                Locale.US, "%.2f", bestScore))

            Toast.makeText(this, resultText, Toast.LENGTH_LONG).show()
            runOnUiThread {
                textViewStatus.text = resultText
                textViewUserName.text = identifiedUserName
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error identifying iris", e)
            Toast.makeText(this, getString(R.string.error_identifying_iris, e.message), Toast.LENGTH_SHORT).show()
        } finally {
            if (irisToIdentify != colorIrisToIdentify) { // Если colorIrisToIdentify был создан как копия
                colorIrisToIdentify.recycle()
            }
            // lastIrisBitmap не очищаем здесь, он обновляется в onCameraFrame
        }
    }

    // --- Activity Lifecycle Methods ---

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initLocal()) {
            Log.e("MainActivity", "OpenCV loader not initialized on resume.")
            // textViewStatus.text = getString(R.string.opencv_initialization_error) // Не перезаписываем статус обработки папки
        } else {
            Log.i("MainActivity", "OpenCV initialized successfully on resume.")
            if (allPermissionsGranted() && (!::eyeClassifier.isInitialized || !::tflite.isInitialized) ){
                initializeOpenCVAndTF() // Инициализируем, если не было сделано ранее
            }
            if (isPreviewRunning && allPermissionsGranted()) {
                openCvCameraView.setCameraIndex(currentCameraId)
                openCvCameraView.enableView()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (openCvCameraView.isVisible) {
            openCvCameraView.disableView()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (openCvCameraView.isVisible) {
            openCvCameraView.disableView()
        }
        if(::tflite.isInitialized) tflite.close()
        lastIrisMat?.release()
        // detectedIrisBitmapFromGallery = null // Удалено
        lastIrisBitmap?.recycle()
        lastIrisBitmap = null
    }
}