package utils

import TrackInFile
import com.google.gson.GsonBuilder
import java.io.File
import java.nio.charset.StandardCharsets

/**
 * Per-file BAT generator for BDRemux pipeline.
 *
 * Contract:
 * - WORKDIR = <src parent>\<basename>
 * - tracks.json is written to WORKDIR\tracks.json
 * - demux.py/verify.py must resolve --tracksData relative to --workdir
 * - audio/sub filenames follow: {trackId}-d{0|1}-{lang}-{trackName}.{ext}
 */
class BdremBatGenerator(
    private val pythonExe: String,
    private val pipelineDir: String,
    private val av1anExe: String = "av1an.exe",
    private val ffmpegExe: String = "ffmpeg",
    private val opusEncExe: String = "opusenc",
) {
    private val gson = GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()

    data class TracksPlan(
        val source: String,
        val workdir: String,
        val tracks: List<TrackPlanEntry>
    )

    data class TrackPlanEntry(
        val trackId: Int,
        val type: String,
        val trackStatus: String,
        val origName: String,
        val origLang: String,
        val trackParam: Map<String, String>,
        val trackMux: Map<String, String>,
        val fileBase: String
    )

    /**
     * Generates:
     * - <src dir>\<basename>.bat
     * - <workdir>\tracks.json
     * - <workdir>\zone_edit_command.txt  (empty placeholder)
     *
     * @param path full path to source mkv/mp4
     * @param tracks per-file tracks list (from GUI result for this file)
     */
    fun generateBat(path: String, tracks: List<TrackInFile>): File {
        val src = File(path)
        require(src.exists() && src.isFile) { "Source file not found: $path" }

        val baseName = src.nameWithoutExtension
        val workDir = File(src.parentFile, baseName)
        val logDir = File(workDir, "00_logs")

        // Create minimal workdir right now (so we can write tracks.json, etc.)
        mkdirsOrThrow(workDir)
        mkdirsOrThrow(logDir)

        // Ensure placeholder command file exists (zone_editor.py often expects it)
        val zoneCmd = File(workDir, "zone_edit_command.txt")
        if (!zoneCmd.exists()) zoneCmd.writeText("", StandardCharsets.UTF_8)

        // Build per-track fileBase for naming stability across demux/audio/mux
        val planEntries = tracks.map { t ->
            TrackPlanEntry(
                trackId = t.trackId,
                type = normalizeType(t.type),
                trackStatus = t.trackStatus.name,
                origName = t.origName,
                origLang = t.origLang,
                trackParam = t.trackParam,
                trackMux = t.trackMux,
                fileBase = buildFileBase(t)
            )
        }

        // Write tracks.json inside WORKDIR (demux/verify resolve relative to --workdir)
        val tracksJsonFile = File(workDir, "tracks.json")
        val plan = TracksPlan(
            source = src.absolutePath,
            workdir = workDir.absolutePath,
            tracks = planEntries
        )
        tracksJsonFile.writeText(gson.toJson(plan), StandardCharsets.UTF_8)

        // Generate BAT next to source file
        val batFile = File(src.parentFile, "$baseName.bat")
        batFile.writeText(buildBatText(src, workDir, logDir, tracks, planEntries), StandardCharsets.UTF_8)
        return batFile
    }

    // ----------------------------
    // BAT building
    // ----------------------------

    private fun buildBatText(
        src: File,
        workDir: File,
        logDir: File,
        tracks: List<TrackInFile>,
        planEntries: List<TrackPlanEntry>
    ): String {
        val sb = StringBuilder()

        fun q(s: String) = "\"$s\""

        val demuxPy = File(pipelineDir, "demux.py").absolutePath
        val attCleanPy = File(pipelineDir, "attachments-cleaner.py").absolutePath
        val autoBoostPy = File(pipelineDir, "auto_boost_2.9.py").absolutePath
        val hdrPatchPy = File(pipelineDir, "hdr_patch.py").absolutePath
        val zoneEditorPy = File(pipelineDir, "zone_editor.py").absolutePath
        val verifyPy = File(pipelineDir, "verify.py").absolutePath
        val muxPy = File(pipelineDir, "mux.py").absolutePath

        // Tracks block
        val tracksComment = buildTracksComment(tracks)

        // Audio EDIT tracks
        val audioEdit = tracks.filter { normalizeType(it.type) == "audio" && it.trackStatus == TrackStatus.EDIT }
        val audioEditBlocks = buildAudioEditBlocks(audioEdit)

        // Video expectations
        val videoEdit = tracks.filter { normalizeType(it.type) == "video" && it.trackStatus == TrackStatus.EDIT }
        require(videoEdit.size == 1) {
            "Expected exactly 1 VIDEO EDIT track for AV1 pipeline; got ${videoEdit.size}. " +
                    "If you want VIDEO COPY support, extend mux.py/contract accordingly."
        }

        sb.appendLine("@echo off")
        sb.appendLine("setlocal EnableExtensions EnableDelayedExpansion")
        sb.appendLine("chcp 65001 >nul")
        sb.appendLine()
        sb.appendLine("set \"WORKERS=8\"")
        sb.appendLine("set \"FASTPASS=--tune 3 --lp 3 --sharpness 1\"")
        sb.appendLine("set \"MAINPASS=--film-grain 10 --complex-hvs 1\"")
        sb.appendLine("set \"AB_MULTIPIER=0.9\"")
        sb.appendLine("set \"AB_POS_DEV=3\"")
        sb.appendLine("set \"AB_NEG_DEV=3\"")
        sb.appendLine()
        sb.appendLine("set \"FASTPASS_VF=\"")
        sb.appendLine("set \"MAINPASS_VF=\"")
        sb.appendLine()
        sb.appendLine("set \"MAINPASS_AUDIO=-an -sn\"")
        sb.appendLine()
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  PER-FILE PIPELINE (generated)")
        sb.appendLine("REM  Source:  ${src.absolutePath}")
        sb.appendLine("REM  Workdir: ${workDir.absolutePath}")
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.appendLine("set \"SRC=${src.absolutePath}\"")
        sb.appendLine("set \"WORKDIR=${workDir.absolutePath}\"")
        sb.appendLine("set \"LOGDIR=%WORKDIR%\\00_logs\"")
        sb.appendLine()
        sb.appendLine("mkdir \"%WORKDIR%\" \"%LOGDIR%\" \"%WORKDIR%\\audio\" \"%WORKDIR%\\video\" \"%WORKDIR%\\sub\" \"%WORKDIR%\\attachments\" \"%WORKDIR%\\chapters\" >nul 2>nul")
        sb.appendLine()
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  1) DEMUX (everything except video)")
        sb.appendLine("REM  Tracks (from GUI):")
        sb.append(tracksComment)
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.appendLine("${pythonExe} ${q(demuxPy)} ^")
        sb.appendLine("  --source \"%SRC%\" ^")
        sb.appendLine("  --workdir \"%WORKDIR%\" ^")
        sb.appendLine("  --tracksData \"tracks.json\" ^")
        sb.appendLine("  > \"%LOGDIR%\\01_demux.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()
        sb.appendLine("REM Optional but recommended")
        sb.appendLine("${pythonExe} ${q(attCleanPy)} ^")
        sb.appendLine("  --subs \"%WORKDIR%\\sub\" ^")
        sb.appendLine("  --attachments \"%WORKDIR%\\attachments\" ^")
        sb.appendLine("  > \"%LOGDIR%\\02_att_clean.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()

        // Video pipeline block (kept close to your preview; generator can be extended later)
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  2) VIDEO (EDIT) -> %WORKDIR%\\video\\video-final.mkv")
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.appendLine("set \"VIDEO_OUT=%WORKDIR%\\video\\video-final.mkv\"")
        sb.appendLine("set \"SCENES=%WORKDIR%\\video\\scenes.json\"")
        sb.appendLine("set \"SCENES_HDR=%WORKDIR%\\video\\scenes-hdr.json\"")
        sb.appendLine("set \"SCENES_FINAL=%WORKDIR%\\video\\scenes-final.json\"")
        sb.appendLine("set \"HDR_WORKDIR=%WORKDIR%\\video\\hdr_tmp\"")
        sb.appendLine("set \"AV1AN_TEMP=%WORKDIR%\\video\\av1an_tmp\"")
        sb.appendLine()
        sb.appendLine("REM 2.1 auto-boost -> scenes.json")
        sb.appendLine("${pythonExe} ${q(autoBoostPy)} ^")
        sb.appendLine("  --source \"%SRC%\" ^")
        sb.appendLine("  --out-scenes \"%SCENES%\" ^")
        sb.appendLine("  --temp \"%AV1AN_TEMP%\\fastpass\" ^")
        sb.appendLine("  --workers \"%WORKERS%\" ^")
        sb.appendLine("  --v-params \"%FASTPASS%\" ^")
        sb.appendLine("  --final-override \"%MAINPASS%\" ^")
        sb.appendLine("  --boost-multiplier \"%AB_MULTIPIER%\" ^")
        sb.appendLine("  --max-pos-dev \"%AB_POS_DEV%\" ^")
        sb.appendLine("  --max-neg-dev \"%AB_NEG_DEV%\" ^")
        sb.appendLine("  --vf \"%FASTPASS_VF%\" ^")
        sb.appendLine("  > \"%LOGDIR%\\03_autoboost.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()
        sb.appendLine("REM 2.2 hdr-patch -> scenes-hdr.json")
        sb.appendLine("${pythonExe} ${q(hdrPatchPy)} ^")
        sb.appendLine("  --source \"%SRC%\" ^")
        sb.appendLine("  --in-scenes \"%SCENES%\" ^")
        sb.appendLine("  --out-scenes \"%SCENES_HDR%\" ^")
        sb.appendLine("  --workdir \"%HDR_WORKDIR%\" ^")
        sb.appendLine("  > \"%LOGDIR%\\04_hdr_patch.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()
        sb.appendLine("REM 2.3 zone-editor -> scenes-final.json")
        sb.appendLine("${pythonExe} ${q(zoneEditorPy)} ^")
        sb.appendLine("  --in \"%SCENES_HDR%\" ^")
        sb.appendLine("  --out \"%SCENES_FINAL%\" ^")
        sb.appendLine("  --command-file \"%WORKDIR%\\zone_edit_command.txt\" ^")
        sb.appendLine("  > \"%LOGDIR%\\05_zone_edit.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()
        sb.appendLine("REM 2.4 mainpass av1an (video only)")
        sb.appendLine("${av1anExe} -i \"%SRC%\" -o \"%VIDEO_OUT%\" ^")
        sb.appendLine("  --scenes \"%SCENES_FINAL%\" ^")
        sb.appendLine("  --workers \"%WORKERS%\" ^")
        sb.appendLine("  --temp \"%AV1AN_TEMP%\\mainpass\" ^")
        sb.appendLine("  --keep ^")
        sb.appendLine("  --resume ^")
        sb.appendLine("  -e svt-av1 ^")
        sb.appendLine("  --pix-format yuv420p10le ^")
        sb.appendLine("  --no-defaults ^")
        sb.appendLine("  -a=\"%MAINPASS_AUDIO%\" ^")
        sb.appendLine("  --vf \"%MAINPASS_VF%\" ^")
        sb.appendLine("  > \"%LOGDIR%\\06_av1an_mainpass.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()

        // Audio EDIT blocks (wav intermediate)
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  3) AUDIO (explicit per-track commands, WAV intermediate)")
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.append(audioEditBlocks)
        sb.appendLine()

        // Subs stage (typically none)
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  4) SUBS (COPY = already present after demux)")
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.appendLine("REM No commands needed here for COPY subs")
        sb.appendLine()

        // Verify stage
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  4.5) VERIFY (integrity check before final mux)")
        sb.appendLine("REM  If fails: creates marker рядом с исходником: <basename>-error-<errorInfo>")
        sb.appendLine("REM  verify.py should write a single-line sanitized errorInfo to: %WORKDIR%\\00_logs\\verify_error.txt")
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.appendLine("${pythonExe} ${q(verifyPy)} ^")
        sb.appendLine("  --source \"%SRC%\" ^")
        sb.appendLine("  --workdir \"%WORKDIR%\" ^")
        sb.appendLine("  --tracksData \"tracks.json\" ^")
        sb.appendLine("  > \"%LOGDIR%\\08_verify.log\" 2>&1")
        sb.appendLine("if errorlevel 1 (")
        sb.appendLine("  set \"ERRINFO=verify_failed\"")
        sb.appendLine("  if exist \"%LOGDIR%\\verify_error.txt\" (")
        sb.appendLine("    for /f \"usebackq delims=\" %%L in (\"%LOGDIR%\\verify_error.txt\") do set \"ERRINFO=%%L\"")
        sb.appendLine("  )")
        sb.appendLine("  call :make_error_marker \"%ERRINFO%\"")
        sb.appendLine("  goto :fail")
        sb.appendLine(")")
        sb.appendLine()

        // Mux stage
        sb.appendLine("REM ==========================================================")
        sb.appendLine("REM  5) MUX -> output рядом с исходником: *-av1.mkv")
        sb.appendLine("REM ==========================================================")
        sb.appendLine()
        sb.appendLine("${pythonExe} ${q(muxPy)} --source \"%SRC%\" --workdir \"%WORKDIR%\" > \"%LOGDIR%\\09_mux.log\" 2>&1")
        sb.appendLine("if errorlevel 1 goto :fail")
        sb.appendLine()
        sb.appendLine("echo OK")
        sb.appendLine("exit /b 0")
        sb.appendLine()
        sb.appendLine(":make_error_marker")
        sb.appendLine("set \"EINFO=%~1\"")
        sb.appendLine("REM sanitize a bit for Windows filename (basic)")
        sb.appendLine("set \"EINFO=%EINFO: =_%\"")
        sb.appendLine("set \"EINFO=%EINFO::=_%\"")
        sb.appendLine("set \"EINFO=%EINFO:/=_%\"")
        sb.appendLine("set \"EINFO=%EINFO:\\=_%\"")
        sb.appendLine("set \"EINFO=%EINFO:*=_%\"")
        sb.appendLine("set \"EINFO=%EINFO:?=_%\"")
        sb.appendLine("set \"EINFO=%EINFO:\\\"=_%\"")
        sb.appendLine("set \"EINFO=%EINFO:< =_%\"")
        sb.appendLine("set \"EINFO=%EINFO:> =_%\"")
        sb.appendLine("set \"EINFO=%EINFO:|=_%\"")
        sb.appendLine("for %%F in (\"%SRC%\") do (")
        sb.appendLine("  set \"SRC_DIR=%%~dpF\"")
        sb.appendLine("  set \"BASENAME=%%~nF\"")
        sb.appendLine(")")
        sb.appendLine("type nul > \"%SRC_DIR%!BASENAME!-error-%EINFO%\"")
        sb.appendLine("exit /b 0")
        sb.appendLine()
        sb.appendLine(":fail")
        sb.appendLine("echo FAILED (code=%errorlevel%)")
        sb.appendLine("echo Logs: \"%LOGDIR%\"")
        sb.appendLine("exit /b 1")

        return sb.toString()
    }

    private fun buildTracksComment(tracks: List<TrackInFile>): String {
        val sb = StringBuilder()
        // Keep stable order: video -> audio -> sub, then by trackId
        val order = mapOf("video" to 0, "audio" to 1, "sub" to 2)
        val sorted = tracks.sortedWith(
            compareBy<TrackInFile> { order[normalizeType(it.type)] ?: 9 }
                .thenBy { it.trackId }
        )

        for (t in sorted) {
            val tType = normalizeType(t.type)
            val status = t.trackStatus.name
            if (tType == "video") {
                sb.appendLine("REM   - VIDEO: ${t.trackId} $status")
                continue
            }
            if (t.trackStatus == TrackStatus.SKIP) {
                sb.appendLine("REM   - ${tType.uppercase()}: ${t.trackId} SKIP")
                continue
            }
            val muxName = (t.trackMux["name"] ?: t.origName).ifBlank { t.origName }
            val muxLang = (t.trackMux["lang"] ?: t.origLang).ifBlank { t.origLang }
            val def = t.trackMux["default"] ?: "false"
            val d = if (def.equals("true", ignoreCase = true)) "d1" else "d0"
            sb.appendLine("REM   - ${tType.uppercase()}: ${t.trackId} $status $d ${muxLang.ifBlank { "und" }} \"${muxName.ifBlank { "-" }}\"")
        }
        return sb.toString()
    }

    private fun buildAudioEditBlocks(audioEdit: List<TrackInFile>): String {
        if (audioEdit.isEmpty()) return "REM No AUDIO EDIT tracks\n"

        val sb = StringBuilder()
        val sorted = audioEdit.sortedBy { it.trackId }
        var logIndex = 7 // after 06_av1an_mainpass
        for (t in sorted) {
            val fileBase = buildFileBase(t)
            val inMka = "%WORKDIR%\\audio\\$fileBase.mka"
            val tmpWav = "%WORKDIR%\\audio\\$fileBase.wav"
            val outOpus = "%WORKDIR%\\audio\\$fileBase.opus"

            val bitrate = parseOpusBitrateKbps(t.trackParam["params"])
            val safeLabel = sanitizeComponentForWindowsFileName(
                (t.trackMux["name"] ?: t.origName).ifBlank { "track${t.trackId}" },
                maxLen = 32
            )

            sb.appendLine("REM 3) Track ${t.trackId} EDIT -> Opus ${bitrate}k (via WAV)")
            sb.appendLine("${ffmpegExe} -v error -y -i \"$inMka\" -map 0:a:0 -vn -sn -c:a pcm_s16le -f wav \"$tmpWav\" ^")
            sb.appendLine("  > \"%LOGDIR%\\0${logIndex}_audio_${t.trackId}_${safeLabel}_wav.log\" 2>&1")
            sb.appendLine("if errorlevel 1 goto :fail")
            logIndex++

            sb.appendLine("${opusEncExe} --quiet --bitrate $bitrate --vbr \"$tmpWav\" \"$outOpus\" ^")
            sb.appendLine("  > \"%LOGDIR%\\0${logIndex}_audio_${t.trackId}_${safeLabel}_opus.log\" 2>&1")
            sb.appendLine("if errorlevel 1 goto :fail")
            logIndex++

            sb.appendLine("del /q \"$tmpWav\" >nul 2>nul")
            sb.appendLine()
        }
        return sb.toString()
    }

    // ----------------------------
    // Naming / parsing helpers
    // ----------------------------

    private fun normalizeType(raw: String): String {
        val v = raw.trim().lowercase()
        return when {
            v.startsWith("vid") || v == "video" -> "video"
            v.startsWith("aud") || v == "audio" -> "audio"
            v.startsWith("sub") || v == "sub" || v == "subtitle" -> "sub"
            else -> v
        }
    }

    private fun buildFileBase(t: TrackInFile): String {
        val def = t.trackMux["default"] ?: "false"
        val d = if (def.equals("true", ignoreCase = true)) "d1" else "d0"

        val langRaw = (t.trackMux["lang"] ?: t.origLang).ifBlank { "und" }
        val nameRaw = (t.trackMux["name"] ?: t.origName).ifBlank { "track${t.trackId}" }

        val lang = sanitizeLang(langRaw)
        val name = sanitizeComponentForWindowsFileName(nameRaw, maxLen = 80)

        return "${t.trackId}-$d-$lang-$name"
    }

    private fun sanitizeLang(lang: String): String {
        // keep short and safe (mkvmerge language tags can be longer, but filename should stay simple)
        val s = lang.trim().ifBlank { "und" }
        val cleaned = s.map { ch ->
            when {
                ch.isLetterOrDigit() -> ch
                ch == '_' || ch == '-' -> ch
                else -> '_'
            }
        }.joinToString("")
        return cleaned.take(16).ifBlank { "und" }
    }

    private fun sanitizeComponentForWindowsFileName(s: String, maxLen: Int): String {
        val bad = charArrayOf('\\', '/', ':', '*', '?', '"', '<', '>', '|')
        var out = s.trim()
        for (c in bad) out = out.replace(c, '_')
        // collapse whitespace
        out = out.replace(Regex("\\s+"), " ").trim()
        // avoid trailing dots/spaces
        out = out.trimEnd('.', ' ')
        if (out.isBlank()) out = "untitled"
        if (out.length > maxLen) out = out.take(maxLen).trimEnd('.', ' ')
        return out
    }

    private fun parseOpusBitrateKbps(params: String?): Int {
        // Accept: "192", "opus 160", "--bitrate 192", etc. First reasonable int wins.
        if (params.isNullOrBlank()) return 192
        val m = Regex("(\\d{2,3})").find(params)
        val v = m?.groupValues?.getOrNull(1)?.toIntOrNull() ?: return 192
        return v.coerceIn(48, 320)
    }

    private fun mkdirsOrThrow(dir: File) {
        if (dir.exists()) return
        if (!dir.mkdirs()) error("Failed to create directory: ${dir.absolutePath}")
    }
}
