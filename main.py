from models.tfidf_logreg import run_tfidf_logreg
from models.tfidf_lsvm import run_tfidf_lsvm

def print_results(name: str, results: dict, split: str):
    print(f"=== {name} ({split}) ===")
    print(f"{split} accuracy: {results['accuracy']:.4f}")
    print(results["report"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print()


def main():
    # dev set metrics
    print_results("TF-IDF + Logistic Regression", run_tfidf_logreg("dev"), "dev")
    print_results("TF-IDF + Linear SVM", run_tfidf_lsvm("dev"), "dev")

    # test set metrics (only uncomment when writing the report)
    # print_results("TF-IDF + Logistic Regression", run_tfidf_logreg("test"), "test")
    # print_results("TF-IDF + Linear SVM", run_tfidf_lsvm("test"), "test")


if __name__ == "__main__":
    main()
