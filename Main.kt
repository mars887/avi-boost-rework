fun main(args: Array<String>) {
    when(args.getOrNull(0)) {
        null -> runMain(args)
        "extract" -> runExtract(args.drop(1))
        "split" -> runSplit(args.drop(1))
        "modify" -> runModify(args.drop(1))
        "pack" -> runPack(args.drop(1))
    }
}

fun runPack(drop: List<String>) {
    TODO("Not yet implemented")
}

fun runModify(drop: List<String>) {
    TODO("Not yet implemented")
}

fun runSplit(drop: List<String>) {
    TODO("Not yet implemented")
}

