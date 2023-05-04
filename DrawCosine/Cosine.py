#画Cosine函数图像
from pylab import *
import numpy as np

#规定取值范围(-pi到pi)
X = np.linspace(-np.pi, np.pi, 256,endpoint=True)

#计算y值
S = np.cos(X)
plot(X,S, color="blue", linewidth=2.5, linestyle="--",label="Cosine")
legend(loc="upper left")

#xlim限制X轴的最小值和最大值，ylim限制Y轴的最小值和最大值。
xlim(X.min()*1.2,X.max()*1.2)
ylim(S.min()*1.2,S.max()*1.2)

#xticks和yticks定义X轴和Y轴坐标刻度和标签，第一个参数为刻度，第二个参数为标签
#ps:这个防止出现两个零点，我们将y轴的零点标签不进行设置
xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
yticks([-1,+1] ,[r'$-1$', r'$+1$'])
ax = gca()

#将右上脊柱设为无色
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#将左下脊柱进行平移
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

#画出箭头
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

#进行描点，画出cos(pi/2)
t=np.pi/2
plot([t,t],[0,np.cos(t)], color ='red', linewidth=2.5, linestyle="--")
scatter([t,],[np.cos(t),], 50, color ='red')

#绘制特殊的点，用annotate函数，第一个参数为显示的内容，xy参数为箭头尖端坐标，xytext参数为文字最左边起始坐标，xycoords坐标系，arrowprops箭头类型
annotate(r'$\cos(\frac{\pi}{2})=0$',
        xy=(t, np.cos(t)),  xycoords='data',
    xytext=(+10, +30), textcoords='offset points',
    fontsize=16,
arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))

#展示图像
show()