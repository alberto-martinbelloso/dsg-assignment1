from algorithms.apriori import apriori
from data_reader import get_data


def main():
    fifa_data = get_data("fifa.csv")
    fifa_data = fifa_data.loc[:, ~fifa_data.columns.str.contains('^Unnamed')]
    shopping_data = get_data("shopping.json")
    telecom_data = get_data("telecom_churn.csv")

    apriori(fifa_data)


if __name__ == "__main__":
    main()
