import com.google.gson.Gson
import utils.BdremBatGenerator
import utils.SummaryRow
import utils.enterNumbers
import utils.summarizeTracksWithPython
import java.io.File

val videoExtensions = listOf("mp4", "mkv")
private val gson = Gson()

fun runMain(args: Array<String>) {
    println("Enter file or directory path:")
    val inputPath = readln().trim()
    val inputFile = File(inputPath)


    val videoFiles = when {
        inputFile.isFile -> if (inputFile.extension in videoExtensions) listOf(inputFile) else error("File extension not supported")

        inputFile.isDirectory -> inputFile.listFiles()?.filter { it.isFile && it.extension in videoExtensions }
            ?.let { files ->
                println("Files:")
                files.forEachIndexed { index, file ->
                    println("  ${index + 1} - ${file.name}")
                }
                val idx = enterNumbers(1, files.size)
                files.filterIndexed { index, _ ->
                    index + 1 in idx
                }
            }

        else -> {
            error("Путь не найден или это не файл/папка.")
        }
    } ?: error("no files found")

    println("Files:")
    println(videoFiles.joinToString("\n") { "  ${it.name}" })

    val tracksSummary = summarizeTracksWithPython(
        Paths.PYTHON_EXE,
        resolveFileTracksInfoPath(),
        videoFiles.map { it.toString() }
    )


    println("\nTracks summary:")
    tracksSummary.forEach {
        println("  ${it.first}")
    }

    val result = openTrackConfigGui(
        pythonExe = Paths.PYTHON_EXE,
        scriptPath = resolveTrackConfigGuiPath(),
        files = videoFiles,
        summary = tracksSummary
    )
    if (result.isEmpty()) {
        println("No GUI result received.")
        return
    }
    result.forEach {
        println(it.key)
        it.value.forEach {
            if(it.trackStatus != TrackStatus.SKIP) {
                println("  ${it.trackId} - ${it.trackStatus}: ${it.trackMux} -> ${it.trackParam}")
            } else {
                println("   ${it.trackId} - ${it.trackStatus}")
            }
        }
    }
    val gen = BdremBatGenerator(
        paths = Paths,
        pipelineDir = "C:\\projects\\PBBatchProcessUtil\\src\\main\\java\\utils"           // где лежат demux.py/mux.py/verify.py/...
    )

    for ((filePath, trackList) in result) {
        val bat = gen.generateBat(filePath, trackList)
        println("Generated: ${bat.absolutePath}")
    }

    val orderedFiles = videoFiles.filter { result.containsKey(it.absolutePath) }
    val batchScripts = gen.generateBatchScripts(orderedFiles)
    batchScripts.forEach {
        println("Generated: ${it.absolutePath}")
    }

}

private fun buildGuiDefaults(): GuiDefaults {
    return GuiDefaults(
        params = "--variance-boost-strength 2 --variance-octile 6 --variance-boost-curve 3 --tune 0 --qm-min 7 --chroma-qm-min 10 --scm 0 --enable-dlf 2 --sharp-tx 1 --enable-restoration 0 --color-primaries 9 --transfer-characteristics 16 --matrix-coefficients 9 --lp 3 --sharpness 1 --hbd-mds 1 --ac-bias 2.00",
        lastParams = "--film-grain 14 --complex-hvs 1",
        zoning = "",
        fastpass = "",
        mainpass = "",
        workers = "8",
        abMultiplier = "0.7",
        abPosDev = "5",
        abNegDev = "4"
    )
}

enum class TrackStatus {
    COPY,
    EDIT,
    SKIP
}

data class TrackInFile(
    val fileIndex: Int,
    val trackId: Int,
    val type: String,
    val origName: String,
    val origLang: String,
    val trackStatus: TrackStatus,
    val trackParam: Map<String, String>,
    val trackMux: Map<String, String>,
)

private fun resolveTrackConfigGuiPath(): String {
    val direct = File(Paths.TRACK_CONFIG_GUI_PY)
    if (direct.exists()) {
        return direct.absolutePath
    }
    val fallback = File("src/main/java/${Paths.TRACK_CONFIG_GUI_PY}")
    if (fallback.exists()) {
        return fallback.absolutePath
    }
    error("track_config_gui.py not found. Checked: ${direct.path}, ${fallback.path}")
}

private fun resolveFileTracksInfoPath(): String {
    val local = File("utils/file-tracks-info.py")
    if (local.exists()) {
        return local.absolutePath
    }
    val fallback = File("src/main/java/utils/file-tracks-info.py")
    if (fallback.exists()) {
        return fallback.absolutePath
    }
    val direct = File(Paths.FILE_TRACKS_INFO_PY)
    if (direct.exists()) {
        return direct.absolutePath
    }
    error(
        "file-tracks-info.py not found. Checked: ${local.path}, ${fallback.path}, ${direct.path}"
    )
}

private fun openTrackConfigGui(
    pythonExe: String,
    scriptPath: String,
    files: List<File>,
    summary: List<Pair<String, SummaryRow>>
): Map<String, List<TrackInFile>> {
    val inputJson = gson.toJson(buildGuiInput(files, summary))
    val cmd = listOf(pythonExe, scriptPath)
    val process = ProcessBuilder(cmd)
        .redirectErrorStream(false)
        .apply {
            environment()["PYTHONIOENCODING"] = "utf-8"
            environment()["PYTHONUTF8"] = "1"
        }
        .start()

    process.outputStream.use { stream ->
        stream.write(inputJson.toByteArray(Charsets.UTF_8))
    }

    val stdout = process.inputStream.bufferedReader(Charsets.UTF_8).readText()
    val stderr = process.errorStream.bufferedReader(Charsets.UTF_8).readText()
    val exitCode = process.waitFor()
    if (exitCode != 0) {
        error("Track config GUI failed with code $exitCode\n$stderr")
    }

    if (stdout.isBlank()) {
        return emptyMap()
    }

    val output = runCatching { gson.fromJson(stdout, GuiOutput::class.java) }
        .getOrElse { error("Invalid JSON output: ${it.message}") }
    val status = output?.status ?: "ok"
    if (status.lowercase() != "ok") {
        return emptyMap()
    }

    val resultAny = output?.result ?: return emptyMap()
    return parseGuiResult(resultAny)
}

private fun buildGuiInput(
    files: List<File>,
    summary: List<Pair<String, SummaryRow>>
): GuiInput {
    val fileList = files.map { it.toString() }
    val summaryRows = summary.map { (line, row) ->
        GuiSummaryRow(
            line = line,
            index = row.index,
            type = row.type,
            displayName = row.displayName,
            lang = row.lang,
            presentIn = row.presentIn.sorted()
        )
    }
    return GuiInput(fileList, summaryRows, buildGuiDefaults())
}



private fun parseGuiResult(resultAny: Map<String, List<GuiTrackEntry>>): Map<String, List<TrackInFile>> {
    val result = mutableMapOf<String, List<TrackInFile>>()
    for ((fileKeyAny, trackListAny) in resultAny) {
        val tracks = trackListAny.map { entry ->
            val status = TrackStatus.valueOf(entry.trackStatus.uppercase())
            TrackInFile(
                entry.fileIndex,
                entry.trackId,
                entry.type ?: "",
                entry.origName ?: "",
                entry.origLang ?: "",
                status,
                entry.trackParam ?: emptyMap(),
                entry.trackMux ?: emptyMap()
            )
        }
        result[fileKeyAny] = tracks
    }
    return result
}

private data class GuiSummaryRow(
    val line: String,
    val index: Int,
    val type: String,
    val displayName: String,
    val lang: String,
    val presentIn: List<Int>
)

private data class GuiDefaults(
    val params: String,
    val lastParams: String,
    val zoning: String,
    val fastpass: String,
    val mainpass: String,
    val workers: String,
    val abMultiplier: String,
    val abPosDev: String,
    val abNegDev: String
)

private data class GuiInput(
    val files: List<String>,
    val summary: List<GuiSummaryRow>,
    val defaults: GuiDefaults
)

private data class GuiOutput(
    val status: String?,
    val result: Map<String, List<GuiTrackEntry>>?
)

private data class GuiTrackEntry(
    val fileIndex: Int,
    val trackId: Int,
    val type: String?,
    val origName: String?,
    val origLang: String?,
    val trackStatus: String,
    val trackParam: Map<String, String>?,
    val trackMux: Map<String, String>?
)
