import com.google.gson.Gson
import utils.BdremBatGenerator
import utils.SummaryRow
import utils.enterNumbers
import utils.summarizeTracksWithPython
import java.io.File
import java.nio.charset.Charset

val videoExtensions = listOf("mp4", "mkv")
private val gson = Gson()

fun runMain(args: Array<String>) {
    println("Enter file, directory path or per-file bat path:")
    val firstInput = readln().trim()
    val firstFile = File(firstInput)

    var perFileDefaults: GuiDefaults? = null
    val inputFile = when {
        firstFile.isFile && firstFile.extension.equals("bat", ignoreCase = true) -> {
            perFileDefaults = runCatching { buildGuiDefaultsFromPerFileBat(firstFile) }
                .onFailure { println("Warning: failed to parse per-file bat: ${it.message}") }
                .getOrNull()
            println("Enter file or directory path:")
            val secondInput = readln().trim()
            File(secondInput)
        }
        else -> firstFile
    }


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
        summary = tracksSummary,
        defaults = perFileDefaults ?: buildGuiDefaults()
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
        pipelineDir = "C:\\projects\\PBBatchProcessUtil\\src\\main\\java"           // где лежат demux.py/mux.py/verify.py/...
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
        sceneDetection = "av1an",
        noFastpass = false,
        fastpassWorkers = "8",
        mainpassWorkers = "8",
        abMultiplier = "0.7",
        abPosDev = "5",
        abNegDev = "4",
        abPosMultiplier = "",
        abNegMultiplier = "",
        mainVpy = "",
        fastVpy = "",
        proxyVpy = ""
    )
}

private fun buildGuiDefaultsFromPerFileBat(batFile: File): GuiDefaults {
    val base = buildGuiDefaults()
    val lines = readBatLines(batFile)
    val env = parseBatEnv(lines)

    val quality = envValue(env, "QUALITY")?.trim()?.takeIf { it.isNotEmpty() }
    val fastPreset = extractFlagValue(lines, "--fast-preset")
    val mainPreset = extractFlagValue(lines, "--preset")
    val noFastpass = lines.any { it.contains("--no-fastpass", ignoreCase = true) }

    val params = buildParamsFromBat(envValue(env, "FASTPASS"), quality, fastPreset)
    val lastParams = buildParamsFromBat(envValue(env, "MAINPASS"), quality, mainPreset)

    val workdir = envValue(env, "WORKDIR")?.takeIf { it.isNotBlank() }
    val zoning = readZoneEdit(workdir, batFile)

    val sceneDetection = envValue(env, "SCENE_DETECTION")
        ?: extractFlagValue(lines, "--sdm")?.takeIf { !it.contains("%") }

    return GuiDefaults(
        params = params ?: base.params,
        lastParams = lastParams ?: base.lastParams,
        zoning = zoning ?: base.zoning,
        fastpass = envValue(env, "FASTPASS_VF") ?: base.fastpass,
        mainpass = envValue(env, "MAINPASS_VF") ?: base.mainpass,
        sceneDetection = sceneDetection ?: base.sceneDetection,
        noFastpass = noFastpass,
        fastpassWorkers = envValue(env, "FASTPASS_WORKERS") ?: base.fastpassWorkers,
        mainpassWorkers = envValue(env, "MAINPASS_WORKERS") ?: base.mainpassWorkers,
        abMultiplier = envValue(env, "AB_MULTIPIER")
            ?: envValue(env, "AB_MULTIPLIER")
            ?: base.abMultiplier,
        abPosDev = envValue(env, "AB_POS_DEV") ?: base.abPosDev,
        abNegDev = envValue(env, "AB_NEG_DEV") ?: base.abNegDev,
        abPosMultiplier = envValue(env, "AB_POS_MULT") ?: base.abPosMultiplier,
        abNegMultiplier = envValue(env, "AB_NEG_MULT") ?: base.abNegMultiplier,
        mainVpy = envValue(env, "MAIN_VPY") ?: base.mainVpy,
        fastVpy = envValue(env, "FAST_VPY") ?: base.fastVpy,
        proxyVpy = envValue(env, "PROXY_VPY") ?: base.proxyVpy
    )
}

private fun readBatLines(batFile: File): List<String> {
    require(batFile.exists() && batFile.isFile) { "Per-file bat not found: ${batFile.path}" }
    val charset = Charset.forName("windows-1251")
    return batFile.readLines(charset)
}

private fun parseBatEnv(lines: List<String>): Map<String, String> {
    val env = mutableMapOf<String, String>()
    val regex = Regex("""^\s*set\s+"([^=]+)=(.*)"\s*$""", RegexOption.IGNORE_CASE)
    for (line in lines) {
        val match = regex.find(line) ?: continue
        val key = match.groupValues[1].trim().uppercase()
        val value = match.groupValues[2]
        env[key] = value
    }
    return env
}

private fun envValue(env: Map<String, String>, key: String): String? {
    val upperKey = key.uppercase()
    return if (env.containsKey(upperKey)) env[upperKey] ?: "" else null
}

private fun extractFlagValue(lines: List<String>, flag: String): String? {
    for (line in lines) {
        val trimmed = line.trimStart()
        if (!trimmed.startsWith(flag)) {
            continue
        }
        val value = extractFlagValue(trimmed, flag)
        if (!value.isNullOrBlank()) {
            return value
        }
    }
    return null
}

private fun extractFlagValue(line: String, flag: String): String? {
    val regex = Regex("""(?:^|\s)${Regex.escape(flag)}\s+("([^"]*)"|([^\s^]+))""")
    val match = regex.find(line) ?: return null
    val quoted = match.groups[2]?.value
    val bare = match.groups[3]?.value
    val value = quoted ?: bare ?: return null
    return value.trim()
}

private fun buildParamsFromBat(base: String?, quality: String?, preset: String?): String? {
    var out = base?.trim() ?: ""
    var used = base != null
    if (!quality.isNullOrBlank() && !hasFlag(out, "--crf")) {
        out = appendFlag(out, "--crf", quality)
        used = true
    }
    if (!preset.isNullOrBlank() && !hasFlag(out, "--preset")) {
        out = appendFlag(out, "--preset", preset)
        used = true
    }
    return if (used) out.trim() else null
}

private fun hasFlag(params: String, flag: String): Boolean {
    if (params.isBlank()) return false
    val regex = Regex("""(?:^|\s)${Regex.escape(flag)}(?:\s|=|$)""")
    return regex.containsMatchIn(params)
}

private fun appendFlag(params: String, flag: String, value: String): String {
    val formatted = formatParamValue(value)
    val prefix = if (params.isBlank()) "" else params.trim() + " "
    return if (formatted.isBlank()) {
        "$prefix$flag"
    } else {
        "$prefix$flag $formatted"
    }.trim()
}

private fun formatParamValue(value: String): String {
    val trimmed = value.trim()
    if (trimmed.isEmpty()) return trimmed
    if (trimmed.startsWith("\"") && trimmed.endsWith("\"")) return trimmed
    return if (trimmed.any { it.isWhitespace() }) "\"$trimmed\"" else trimmed
}

private fun readZoneEdit(workdir: String?, batFile: File): String? {
    val candidates = mutableListOf<File>()
    if (!workdir.isNullOrBlank()) {
        candidates.add(File(workdir, "zone_edit_command.txt"))
    }
    val fallbackDir = File(batFile.parentFile, batFile.nameWithoutExtension)
    candidates.add(File(fallbackDir, "zone_edit_command.txt"))

    for (candidate in candidates) {
        if (candidate.exists() && candidate.isFile) {
            return candidate.readText(Charsets.UTF_8)
        }
    }
    return null
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
    summary: List<Pair<String, SummaryRow>>,
    defaults: GuiDefaults
): Map<String, List<TrackInFile>> {
    val inputJson = gson.toJson(buildGuiInput(files, summary, defaults))
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
    summary: List<Pair<String, SummaryRow>>,
    defaults: GuiDefaults
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
    return GuiInput(fileList, summaryRows, defaults)
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
    val sceneDetection: String,
    val noFastpass: Boolean,
    val fastpassWorkers: String,
    val mainpassWorkers: String,
    val abMultiplier: String,
    val abPosDev: String,
    val abNegDev: String,
    val abPosMultiplier: String,
    val abNegMultiplier: String,
    val mainVpy: String,
    val fastVpy: String,
    val proxyVpy: String
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
