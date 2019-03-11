from algorithms.apriori import apriori
from data_reader import get_data


def main():
    fifa_data = get_data("fifa.csv")
    shopping_data = get_data("shopping.json")
    telecom_data = get_data("telecom_churn.csv")

    apriori_result = apriori(shopping_data)


if __name__ == "__main__":
    main()
