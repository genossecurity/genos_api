from stress import run_offline_audit


def run_integrated_audit():
    run_offline_audit(sample_size=1000)


if __name__ == "__main__":
    run_integrated_audit()