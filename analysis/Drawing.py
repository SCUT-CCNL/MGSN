import matplotlib.pyplot as plt


def auto_label(bar):
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")


def diagram():
    plt.figure(figsize=(4, 3), dpi=300)
    plt.rc('font', family='Times New Roman', size=10)
    name = ['[1,6)', '[6,11)', '>11']
    value1 = [53.1, 66.8, 73.2]
    value2 = [53.3, 63.9, 75.2]
    value3 = [55.9, 66.2, 76.7]

    bar1 = plt.bar([i - 0.25 for i in range(3)], height=value1, width=0.2, alpha=0.8, hatch='/', color='lightslategrey', label='BiLSTM(+SciBERT)')
    bar2 = plt.bar([i for i in range(3)], height=value2, width=0.2, alpha=0.8, hatch='\\', color='lightskyblue', label='BRAN(+SciBERT)')
    bar3 = plt.bar([i + 0.25 for i in range(3)], height=value3, width=0.2, alpha=0.8, hatch='x', color='lightgreen', label='MGSN')

    plt.xticks(range(3), name)
    plt.yticks(range(30, 105, 15))
    plt.xlabel('number of target entity pairs')
    plt.ylabel('F1 (%)')

    plt.legend(loc='upper left', frameon=False, fontsize=6)

    auto_label(bar1)
    auto_label(bar2)
    auto_label(bar3)
    plt.savefig('./test.png', dpi=1200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    diagram()