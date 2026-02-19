from models.tfidf_logreg import run_tfidf_logreg
from models.tfidf_lsvm import run_tfidf_lsvm

def main():
    logreg_results = run_tfidf_logreg()

    print("=== TF-IDF + Logistic Regression ===")
    print(f"Dev accuracy: {logreg_results['accuracy']:.4f}")
    print(logreg_results["report"])
    print("Confusion matrix:")
    print(logreg_results["confusion_matrix"])

    lsvm_results = run_tfidf_lsvm()

    print("=== TF-IDF + Linear SVM ===")
    print(f"Dev accuracy: {lsvm_results['accuracy']:.4f}")
    print(lsvm_results["report"])
    print("Confusion matrix:")
    print(lsvm_results["confusion_matrix"])


if __name__ == "__main__":
    main()
