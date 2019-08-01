import pandas
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter


def draw_result(lst_iter, lst_loss, lst_acc, title):

    lst_acc = lst_acc.transform(lambda x: (1 - x))

    print(lst_acc)

    formatter = FuncFormatter(lambda y, _: '{:,.2%}'.format(y))

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)

    #plt.plot(lst_iter, lst_loss, '-b', label='Perda em treinamento')
    plt.plot(lst_iter, lst_acc, '-r', label='Acurácia')

    plt.xlabel("Épocas")
    plt.legend(loc='lower right')
    #plt.title(title)

    plt.xticks(np.arange(min(lst_iter), max(lst_iter) + 5, 5.0))

    # save image
    plt.savefig(title+'.png', bbox_inches='tight', dpi=1000)  # should before show method

    # show
    plt.show()

def read_perf_file():

    csv = pandas.read_csv("./perf/pan-seq-perf.csv", header=None)

    return csv

def test_draw():
    # iteration num
    lst_iter = range(100)

    content = read_perf_file()

    # loss of iteration
    lst_loss = content[2]
    # lst_loss = np.random.randn(1, 100).reshape((100, ))

    # accuracy of iteration
    lst_acc = content[4]
    # lst_acc = np.random.randn(1, 100).reshape((100, ))
    draw_result(lst_iter, lst_loss, lst_acc, "Experimento")


if __name__ == '__main__':
    test_draw()