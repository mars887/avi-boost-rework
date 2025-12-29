package utils

import kotlin.math.max
import kotlin.math.min

fun List<String>.toRange(): IntRange {
    val nums = take(2).mapNotNull { it.toIntOrNull() }
    return min(nums[0], nums[1])..max(nums[0], nums[1])
}

fun enterNumbers(min: Int, max: Int): List<Int> {
    val lst = mutableListOf<Int>()
    val dat = readln()
    var errExp: Pair<String, String?>? = null

    dat.split(" ").forEach {
        try {
            errExp = Pair(it,null)
            when {
                it == "*" -> {
                    errExp = Pair(errExp.first,"*")
                    (min..max).forEach { lst += it }
                }

                it.contains("..") -> {
                    errExp = Pair(errExp.first,"..")
                    val range = it.split("..").toRange()
                    range.forEach { lst += it }
                }

                it.contains("-.") -> {
                    errExp = Pair(errExp.first,"-.")
                    val range = it.split("-.").toRange()
                    range.forEach { lst -= it }
                }

                it.toIntOrNull() != null -> {
                    errExp = Pair(errExp.first,"")
                    lst += it.toInt()
                }

                it.drop(1).toIntOrNull() != null && it.first() == '/' -> {
                    errExp = Pair(errExp.first,"/")
                    lst -= it.drop(1).toInt()
                }
            }
        } catch (e: Exception) {
            println("[W] error on token \'${errExp?.first}\', exp - \'${errExp?.second}\', token skipped, error message: ${e.message}")
            e.printStackTrace()
        }
    }
    return lst.filter { it in min..max }
}
