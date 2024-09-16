## Hinweise zur Softwareentwicklung !!!

# 1. Entwicklungsumbung in conda aktivieren!
# conda activate env311

chimcla_step_history_eval -l classifier-2023-07-10_since_2023-06-26.log -cm bilder1/stage3_results__history_test_y\* 300

# change directory
cd workdata

# copy selected images

cp 2024-07-* ../test        



# plot diagram

x = df_sum.dt_abs

In [2]: type(x)
Out[2]: pandas.core.series.Series

In [3]: x.to_numpy()
Out[3]: array([277.37, 277.42, 275.88, ..., 688.55, 692.47, 697.45])

In [4]: x = df_sum.dt_abs.to_numpy

In [5]: x = df_sum.dt_abs.to_numpy()

In [6]: plt.plot(x)
Out[6]: [<matplotlib.lines.Line2D at 0x1310eb350>]

In [7]: plt.show()

In [8]: y = df_sum.dt_abs

In [9]: y.plot?

In [10]: df_sum["dt_abs"].plot(kind = 'bar', y = 'dwell_time', use_index = False)
Out[10]: <Axes: >

In [11]: plt.show()

In [12]: 8+8
Out[12]: 16

In [13]: df_sum["dt_abs"].plot(kind = 'bar', y = 'dwell_time', xticks = range(0,1500,100))
Out[13]: <Axes: >

In [14]: plt.show()