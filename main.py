from algorithms.apriori import apriori
from data_reader import get_data
import pandas as pd
import matplotlib.pyplot as plt


def run_algorithm(algorithm):

    if algorithm == 'apriori':
        data = get_data("shopping.json")
        apriori_result = apriori(data, 0.3, 0.5)
        apriori_result = apriori_result.sort_values(by=['ir'], ascending=[True])

        plt.show(apriori_result.query('ir > 0.0')
                 .plot(kind='bar', x='rule', y=['ir', 'kulczynski'], rot=45, fontsize=6))

        print(apriori_result)


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)

    run_algorithm('apriori')


if __name__ == "__main__":
    main()
