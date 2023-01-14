//val command = Array(
//  "--class",
//  "org.apache.spark.repl.Main",
//  "spark-shell")
//org.apache.spark.deploy.SparkSubmit.main(command)

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import CG.cg
import scala.util.control.Breaks._
import scala.math._
import java.io._

object NPSVM {
  type Vector = DenseVector[Double]
  type IntVector = DenseVector[Int]
  type Matrix = DenseMatrix[Double]
  type DMat = RDD[Matrix]
  type DFitted = (Vector, Vector, Vector, Vector)

  //可以先转化为矩阵，然后再划分分区
  //将T,Q两个矩阵读入，分布式计算节约T'T，Q’Q资源
  def Read(file:String,spark: SparkSession,num_part: Int):DMat={
    var M = spark.read.format("csv").option("header", "true").load(file)
    M = M.repartition(num_part)
    println(M.rdd.getNumPartitions)

    val part2m = (part: Iterator[Row]) => {
    val rows = part.map(row => row.toSeq.toArray.map(x => x.toString.toDouble)).toArray
    // https://stackoverflow.com/a/48168079
    //val xarr = rows.map(row => 1.0 +: row.drop(1))
    DenseMatrix(rows: _*)
  }
    val dmat = M.rdd.mapPartitions(part_iter => Iterator[DenseMatrix[Double]](part2m(part_iter)))
    dmat

  }

  //声明变量C和z,U0,L0
  def generate(epsilon:Double,c:Double,target:Int,other:Int):DFitted={
    val C=DenseVector.ones[Double](2*target+other)*c
    var z=DenseVector.ones[Double](2*target+other)
    for (i <- 0 to 2*target-1){
      z(i)=(-1.0)*z(i)*epsilon
    }
    val L0=DenseVector.ones[Double](2*target+other)
    val U0=DenseVector.ones[Double](2*target+other)
    (C,z,L0,U0)
  }

  //更新P
  // 定义矩阵 A = rho*X'X + I 及其运算 x => A * x
  //
  def MtM_prod(part: DMat,v:Vector): Vector = {
    val dxtxv = part.map(x => x.t * (x*v))
    val xtxv = dxtxv.reduce(_ + _)
    xtxv
  }
  def Aprod(v:Vector, args: Any): Vector = {
    val (xx, ho)=args.asInstanceOf[(DMat,Double)]
    val res = MtM_prod(xx,v)
    val res1=res * ho
    res1 :+= v
    res1
  }
  def P_update(TT:DMat,Rho:Double,lold:Vector,Z:Vector,uold:Vector): Vector ={
    val dbeta=TT.map(x=>(-1.0)*Rho*x.t*(lold-Z+uold))
    val d=TT.first().cols //矩阵的维度
    val beta=dbeta.reduce(_ + _)
    val eps = 1e-6
    val Con=(TT,Rho)
    val Pnew=cg(d,Aop=Aprod,args=Con,b=beta,eps)
    Pnew
  }

  //更新t
  def t_update(Z:Vector,T:DMat,Pnew:Vector,uold:Vector): Vector ={
    val TPd=T.map(x=>x*Pnew)
    val TP=TPd.reduce(_+_)
    val tnew=Z-TP-uold
    tnew
  }

  //更新L
  def L_update(tnew:Vector,CC:Vector,Rho:Double):Vector={
    val j=tnew.length
    val tr=CC/Rho
    var Lnew=DenseVector.zeros[Double](j)
    for (i<-0 to j-1){
      if(tnew(i)>tr(i)){
        Lnew(i)=tnew(i)-tr(i)
      }
      if(tnew(i)>0 && tnew(i)<tr(i)){
        Lnew(i)=tnew(i)+tr(i)
      }
    }
    Lnew
  }

  //更新U
  def U_update(uold:Vector,T:DMat,Pnew:Vector,Z:Vector):Vector={
    val TPd=T.map(x=>x*Pnew)
    val TP=TPd.reduce(_+_)
    val Unew=uold+TP-Z
    Unew
  }

  //迭代求解
  def P_final(TQ:DMat,Max:Int,S:DFitted,Rho:Double,target:Int,other:Int):Vector={
    val C=S._1
    val z=S._2
    var L=S._3
    var U=S._4
    val P=DenseVector.zeros[Double](TQ.first().cols)
    var i=0
    val ABSTOL   = 1e-4
    val RELTOL   = 1e-2
    var rval=DenseVector.zeros[Double](Max)
    var sval=DenseVector.zeros[Double](Max)
    var eps_pri=DenseVector.zeros[Double](Max)
    var eps_dual=DenseVector.zeros[Double](Max)
    breakable{
      i=i+1
      val Lold=L.toDenseVector
      val Uold=U.toDenseVector
      val P=P_update(TQ,Rho,Lold,z,Uold)
      val t=t_update(z,TQ,P,Uold)
      L=L_update(t,C,Rho)
      U=U_update(Uold,TQ,P,z)
      val TQPd=TQ.map(x=>x*P)
      val TQP=TQPd.reduce(_+_)
      val r=TQP+L-z
      val Lup=L-Lold
      val sd=TQ.map(x=>x.t * Lup*Rho)
      val s=sd.reduce(_+_)
      rval(i)=norm(r)
      sval(i)=norm(s)
      val TQtUd=TQ.map(x=>x.t * U)
      val TQtU=TQtUd.reduce(_+_)
      eps_pri(i) = math.sqrt(2*target+other+2)*ABSTOL + RELTOL*math.max(math.max(norm(TQP),norm(L)),norm(z))
      eps_dual(i)= math.sqrt(2*(2*target+other))*ABSTOL + RELTOL*norm(Rho*TQtU)
      if  (rval(i) < eps_pri(i) && sval(i) < eps_dual(i))  break
    }
    P
  }


  def main(args: Array[String]): Unit ={
    //读取数据,分区
    val spark = SparkSession.builder().appName("NPSVM").getOrCreate()
    // 预览数据前5行
    spark.sparkContext.textFile("E:/final_data.csv").take(5).foreach(println)
    // 读取数据
    var mat = spark.read.format("csv").option("header", "true").load("E:/final_data.csv")
    //返回样本数量
    val N=mat.count().toInt   //N=p+q checked 实验数据量为30686
    //对数据进行筛选
    val pos=mat.where("y=1")
    val p=pos.count().toInt   //正例数量
    val neg=mat.where("y=0")
    val q=neg.count().toInt  //负例数量
    //整理结果，合并结果
    val pathP="E:/positive.csv"
    val pathN="E:/negative.csv"
    val num=150
    val T=Read(pathP,spark,num)
    val Q=Read(pathN,spark,num)
    val epsilon=1e-5
    val c=1
    val rho=1.0
    val thr=c/rho
    val MAX_ITER=1000
    val S1=generate(1e-5,1.0,p,q)
    val P_pos=P_final(T,MAX_ITER,S1,rho,p,q)
    val S2=generate(1e-5,1.0,q,p)
    val P_neg=P_final(Q,MAX_ITER,S2,rho,q,p)
    //把pos和neg变成矩阵进行预测
    //去掉第一列，只保留x
    val part2mat = (part: Iterator[Row]) => {
      val rows = part.map(row => row.toSeq.toArray.map(x => x.toString.toDouble)).toArray
      val x=rows.map(c=>c.drop(1))
      // https://stackoverflow.com/a/48168079
      val xx=DenseMatrix(x: _*)
      xx
    }
    val Pos = pos.rdd.mapPartitions(part_iter => Iterator[DenseMatrix[Double]](part2mat(part_iter)))
    val Pos1=Pos.first()
    val Neg = neg.rdd.mapPartitions(part_iter => Iterator[DenseMatrix[Double]](part2mat(part_iter)))
    val Neg1=Neg.first()
    //根据得到的两组w和b看精度
    val w1=P_pos(0,P_pos.length-1)
    val b1=P_pos(-1)
    val w2=P_neg(0,P_neg.length-1)
    val b2=P_neg(-1)
    var sv1=0
    var sv2=0
    val R1=Pos1*w1+b1
    val R2=Pos1*w2+b2
    val R3=Neg1*w1+b1
    val R4=Neg1*w2+b2
    //正
    for (i <- 0 to (p-1) ){
      if (R1(i)>=epsilon && R2(i)<=1) {
        sv1=sv1+1
      }
    }
    //负
    for (j <- 0 to (q-1) ){
      if (R3(j)>=epsilon && R4(j)<=1) {
        sv2=sv2+1
      }
    }
    val acc=(p+q-sv1-sv2)/(p+q)
    println(s"Accuracy of NPSVM is $acc ")
    spark.stop()
  }

}
