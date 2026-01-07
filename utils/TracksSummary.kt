package utils

import java.io.BufferedReader
import kotlin.collections.forEach

data class RawTrack(
    val fileIndex: Int,   // 1-based file index
    val ffIndex: Int,     // ffprobe stream index
    val type: String,     // "Video", "Audio", "Subtitle", ...
    val title: String,
    val language: String,
)

// Row of final summary
data class SummaryRow(
    val index: Int,
    val star: Boolean,
    val type: String,
    val displayName: String,
    val lang: String,
    val presentIn: Set<Int>,   // files where this variant exists
    var reason: String = ""    // will be filled later: "", "(except ...)", "-> (only ...)"
)

fun summarizeTracksWithPython(
    pythonExe: String,
    scriptPath: String,
    files: List<String>
): List<Pair<String, SummaryRow>> {
    require(files.isNotEmpty()) { "Files list must not be empty" }

    // 1. Run python script
    val cmd = mutableListOf(pythonExe, scriptPath).apply {
        addAll(files)
    }

    val process = ProcessBuilder(cmd)
        .redirectErrorStream(true)
        .apply {
            environment()["PYTHONIOENCODING"] = "utf-8"
            environment()["PYTHONUTF8"] = "1"
        }
        .start()

    val output = process.inputStream.bufferedReader(Charsets.UTF_8).use(BufferedReader::readText)
    val exitCode = process.waitFor()
    if (exitCode != 0) {
        throw IllegalStateException("Python exited with code $exitCode\n$output")
    }

    // 2. Parse track lines from output of file-tracks-info.py
    val tracks = parseTracksFromScriptOutput(output, files.size)

    // 3. Build summary according to your rules
    return buildSummary(files.size, tracks)
}

fun parseTracksFromScriptOutput(
    output: String,
    filesCount: Int
): List<RawTrack> {
    val result = mutableListOf<RawTrack>()
    var currentFileIndex = 0 // 0-based, will store 1-based

    output.lineSequence().forEach { rawLine ->
        val line = rawLine.trim()
        if (line.isEmpty()) return@forEach

        if (line.startsWith("===")) {
            // New file marker: "=== path ==="
            currentFileIndex += 1
            return@forEach
        }

        // Skip lines that obviously are not stream rows
        if (!line.contains("|")) return@forEach

        // Split by " | "
        val parts = line.split("|").map { it.trim() }
        if (parts.size < 4) return@forEach

        val type = parts[0]                  // "Video" | "Audio" | "Subtitle" ...
        val idx = parts[1].toIntOrNull() ?: return@forEach
        val title = parts[2]
        val lang = parts[3]

        if (currentFileIndex !in 1..filesCount) {
            // Something is wrong, but silently skip
            return@forEach
        }

        if(type != "attachment") result += RawTrack(
            fileIndex = currentFileIndex, // make it 1-based
            ffIndex = idx,
            type = type,
            title = title,
            language = lang
        )
    }

    return result
}

// Helper for final formatting
fun formatFileSet(files: Set<Int>): String {
    if (files.isEmpty()) return ""
    val sorted = files.sorted()
    val ranges = mutableListOf<IntRange>()

    var start = sorted[0]
    var end = sorted[0]

    for (i in 1 until sorted.size) {
        val n = sorted[i]
        if (n == end + 1) {
            end = n
        } else {
            ranges += start..end
            start = n
            end = n
        }
    }
    ranges += start..end

    return ranges.joinToString(", ") { r ->
        if (r.first == r.last) r.first.toString() else "${r.first}-${r.last}"
    }
}

fun buildSummary(
    filesCount: Int,
    tracks: List<RawTrack>
): List<Pair<String, SummaryRow>> {
    if (tracks.isEmpty()) return emptyList()

    // How we identify tracks logically (per type + "name")
    fun identFor(t: RawTrack): String =
        if (t.type == "Video") t.language else t.title

    data class TrackKey(val type: String, val ident: String, val lang: String)

    // Build trackKey -> (fileIndex -> ffIndex)
    val trackMap = mutableMapOf<TrackKey, MutableMap<Int, Int>>()
    for (t in tracks) {
        val key = TrackKey(t.type, identFor(t), t.language)
        val byFile = trackMap.getOrPut(key) { mutableMapOf() }
        byFile[t.fileIndex] = t.ffIndex
    }

    // Anchored tracks: present in ALL files and always at the same index.
    // Shifting tracks: present in >=2 files but at different indices.
    val anchoredIndex = mutableMapOf<TrackKey, Int>()
    val shiftingKeys = mutableSetOf<TrackKey>()

    for ((key, byFile) in trackMap) {
        val idxSet = byFile.values.toSet()
        if (byFile.size == filesCount && idxSet.size == 1) {
            anchoredIndex[key] = idxSet.first()
        } else if (byFile.size >= 2 && idxSet.size > 1) {
            shiftingKeys += key
        }
    }

    val shiftingIndexMap: Map<TrackKey, Set<Int>> =
        shiftingKeys.associateWith { key -> trackMap[key]!!.values.toSet() }

    // Group tracks by ffIndex
    val tracksByIndex: Map<Int, List<RawTrack>> = tracks.groupBy { it.ffIndex }
    val allIndexes = tracksByIndex.keys.sorted()

    // Decide which indexes are starred
    val starByIndex = mutableMapOf<Int, Boolean>()
    for (idx in allIndexes) {
        val atIndex = tracksByIndex[idx]!!
        val filesAtIndex = atIndex.map { it.fileIndex }.toSet()
        val typesAtIndex = atIndex.map { it.type }.toSet()

        val hasAnchor = anchoredIndex.values.any { it == idx }
        val hasShiftedHere = shiftingIndexMap.values.any { idx in it }

        val safe = when {
            // Perfectly aligned by a concrete track (type+name/title)
            hasAnchor -> true

            // All files have something at this index, all of same type,
            // and no "shifting" track passes through this index.
            filesAtIndex.size == filesCount &&
                    typesAtIndex.size == 1 &&
                    !hasShiftedHere -> true

            else -> false
        }

        starByIndex[idx] = !safe
    }


    // Build raw rows (per index, per (type, displayName))
    val rows = mutableListOf<SummaryRow>()
    for ((idx, atIndex) in tracksByIndex) {
        val star = starByIndex[idx] ?: true

        val groups = mutableMapOf<TrackKey, MutableSet<Int>>()
        for (t in atIndex) {
            val displayName = identFor(t)
            val key = TrackKey(t.type, displayName, t.language)
            groups.getOrPut(key) { mutableSetOf() } += t.fileIndex
        }

        for ((key, filesSet) in groups) {
            rows += SummaryRow(
                index = idx,
                star = star,
                type = key.type,
                displayName = key.ident,
                lang = key.lang,
                presentIn = filesSet.toSortedSet()
            )
        }
    }

    // Fill reasons (only / except)
    val rowsByIndex = rows.groupBy { it.index }
    val majorityThreshold = 0.6

    for ((_, idxRows) in rowsByIndex) {
        val starIndex = idxRows.first().star

        if (starIndex) {
            // For starred indexes everything that is not in all files is "only"
            idxRows.forEach { row ->
                if (row.presentIn.size < filesCount) {
                    row.reason = "-> (only ${formatFileSet(row.presentIn)})"
                }
            }
        } else {
            if (idxRows.size == 1) {
                val row = idxRows.first()
                if (row.presentIn.size < filesCount) {
                    row.reason = "-> (only ${formatFileSet(row.presentIn)})"
                }
            } else {
                // Multiple name variants at the same index
                val majorityRows = idxRows.filter { row ->
                    val frac = row.presentIn.size.toDouble() / filesCount
                    row.presentIn.size >= 2 && frac >= majorityThreshold
                }

                if (majorityRows.isNotEmpty()) {
                    // Majority rows -> "except", others -> "only"
                    majorityRows.forEach { row ->
                        val missing = (1..filesCount).filter { it !in row.presentIn }.toSet()
                        if (missing.isNotEmpty()) {
                            row.reason = "-> (except ${formatFileSet(missing)})"
                        }
                    }
                    idxRows.filter { it !in majorityRows }.forEach { row ->
                        row.reason = "-> (only ${formatFileSet(row.presentIn)})"
                    }
                } else {
                    // No majority: everything is "only"
                    idxRows.forEach { row ->
                        row.reason = "-> (only ${formatFileSet(row.presentIn)})"
                    }
                }
            }
        }
    }

    // Final pretty-print
    val result = mutableListOf<Pair<String, SummaryRow>>()
    rowsByIndex.toSortedMap().forEach { (_, idxRows) ->
        val sortedRows = idxRows.sortedWith(
            compareBy<SummaryRow> { it.index }
                .thenBy { it.type }
                .thenBy { it.displayName }
        )

        for (row in sortedRows) {
            val starPrefix = if (row.star) "*" else ""
            val base = "$starPrefix${row.index} | ${row.type} | ${row.displayName} | ${row.lang}"
            val line = if (row.reason.isNotBlank()) {
                "$base ${row.reason}"
            } else {
                base
            }
            result += Pair(line,row)
        }
    }

    return result
}
