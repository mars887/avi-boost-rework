import utils.enterNumbers
import utils.summarizeTracksWithPython
import java.io.File

val videoExtensions = listOf("mp4", "mkv")

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
        Paths.FILE_TRACKS_INFO_PY,
        videoFiles.map { it.toString() }
    )

    println("\nTracks summary:")
    tracksSummary.forEach {
        println("  ${it.first}")
    }


    val result = mutableMapOf<String, List<TrackInFile>>
    // open config gui
}

data class TrackInFile(
    val fileIndex: Int,
    val trackId: Int,
    val trackStatus: TrackStatus,
    val trackParam: Map<String, String>,
    val trackMux: Map<String, String>,
)
