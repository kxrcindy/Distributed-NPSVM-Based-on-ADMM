// 载入 Breeze 的线性代数库
import breeze.linalg._
import breeze.stats.distributions._

import scala.util.control.Breaks._

// 求解 Ax=b，A 为正定矩阵
object CG extends App {
  // 设置随机数种子
  val seed = 123456
  implicit val randBasis: RandBasis = RandBasis.withSeed(seed)

  // 设置矩阵维度
  val m = 1000

  // 创建分布对象
  val normal = new Gaussian(0.0, 1.0)
  // 生成正定矩阵 A
  val A0 = new DenseMatrix(2 * m, m, normal.sample(2 * m * m).toArray)
  val A = A0.t * A0
  // 生成向量 b
  val b = new DenseVector(normal.sample(m).toArray)
  // 真实解
  val xtrue = A \ b

  // 定义运算 x => A * x
  type Vector = DenseVector[Double]
  type Matrix = DenseMatrix[Double]
  def Aprod(x: Vector, args: Any): Vector = {
    val A = args.asInstanceOf[Matrix]
    A * x
  }

  // 利用 CG 求解，并与真实值对比
  // eps 用来指定残差向量 r 的容忍度，||r|| <= eps 时即可退出迭代
  def cg(m: Int, Aop: (Vector, Any) => Vector, args: Any, b: Vector, eps: Double = 0.001): Vector = {
    // 初值
    val x = DenseVector.zeros[Double](m)

    // 完成函数编写
    val r = b - Aop(x, args)
    val p = r.toDenseVector  // 复制整个向量

    breakable {
      for (i <- 1 to m) {
        val Ap = Aop(p, args)
        val alpha = (r dot r) / (p dot Ap)
        x :+= alpha * p
        val rnew = r - alpha * Ap
        if (norm(rnew) <= eps) break
        println(s"iter $i, resid = ${norm(rnew)}")
        val beta = (rnew dot rnew) / (r dot r)
        p := rnew + beta * p
        r := rnew
      }
    }

    x
  }

  val xcg = cg(m, Aop = Aprod, args = A, b = b, eps = 1e-6)
  println(s"xtrue = $xtrue")
  println(s"xcg = $xcg")

  // 定义矩阵 B = diag(d1, d2, ..., dm)，利用 CG 求解 Bx=b，并与真实值对比
  // (d1, ..., dm) = (1, 2, ..., m)
}
