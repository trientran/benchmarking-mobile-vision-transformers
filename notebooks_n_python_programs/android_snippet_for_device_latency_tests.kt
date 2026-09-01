   private suspend fun labelImages(addedUris: List<Uri>) {
        val batteryLevel = getBatteryPercentage()
        Timber.d("trien Current battery level: $batteryLevel%")

        // Warmup
        repeat(100000) { repetition ->
            val entireBitmap = getBitmapFromFileUri(addedUris[0], MAX_IMAGE_DIMENSION_FOR_LABELING) ?: return
            val inputImage = InputImage.fromBitmap(entireBitmap, 0)
            labelSingleImage(inputImage) {
                Timber.v("trien warm up ${it.first().text}")
                Timber.v("trien warm up ${repetition}")
                if (repetition % 100 == 0) {
                    val batteryLevel = getBatteryPercentage()
                    Timber.d("trien Current battery level of recognition: $batteryLevel%")
                }
            }
        }

        addedUris.onEachIndexed { _, uri ->
            val start = System.nanoTime()
            val entireBitmap = getBitmapFromFileUri(uri, MAX_IMAGE_DIMENSION_FOR_LABELING) ?: return
            val inputImage = InputImage.fromBitmap(entireBitmap, 0)
            labelSingleImage(inputImage) { labels ->
                val currentRecognitionList = state.recognitionList.toMutableList()
                if (labels.isEmpty()) {
                    currentRecognitionList.add(Recognition(fileUri = uri, herbs = emptyList()))
                } else {
                    val maxResultsDisplayed = labels.size
                    val herbs = mutableListOf<Herb>()
                    for (i in 0 until maxResultsDisplayed) {
                        val id = labels[i].text
                        herbs.add(
                            Herb(
                                id = id,
                                latinName = state.recognizedLatinHerbs!!.getString(id),
                                viName = state.recognizedViHerbs!!.getString(id),
                                confidence = labels[i].confidence
                            )
                        )
                    }
                    currentRecognitionList.add(Recognition(fileUri = uri, herbs = herbs))
                }
                setState { copy(recognitionList = currentRecognitionList) }

                // Timed run
                val latencies = mutableListOf<Long>()
                latencies.add(System.nanoTime() - start)
                if (uri == addedUris.last()) {
                    val medianMs = latencies.sorted()[latencies.size / 2] / 1_000_000.0
                    Timber.tag("BENCH").i("Median latency: $medianMs ms")
                }
            }
        }
    }

    private fun getBatteryPercentage(): Int {
        val batteryManager = application.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        return batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
    }
