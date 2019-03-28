from algorithms.apriori import apriori
from data_reader import get_data
import pandas as pd
import matplotlib.pyplot as plt


def main():
    fifa_data = get_data("fifa.csv")
    shopping_data = get_data("shopping.json")
    telecom_data = get_data("telecom_churn.csv")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)

    apriori_result = apriori(shopping_data, 0.15, 0.3)
    # apriori_result = apriori_result.sort_values(by=['confidence'], ascending=[True])
    apriori_result = apriori_result.sort_values(by=['ir'], ascending=[True])
    print(apriori_result)

    plt.show(apriori_result.query('ir > 0.0')
             .plot(kind='bar', x='rule', y=['ir', 'kulczynski'], rot=45, fontsize=6))

    # plt.show(apriori_result.query('ir > 0.0')
    #          .plot(kind='bar', x='rule', y=['support', 'confidence'], rot=45, fontsize=6))


if __name__ == "__main__":
    main()
