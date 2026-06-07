# EVALUATE MODEL

def evaluate_model(model, test_ds):

    print("\nEvaluating model...\n")

    test_loss, test_acc = model.evaluate(
        test_ds
    )

    print(f"\nTest Loss: {test_loss:.4f}")

    print(f"Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc