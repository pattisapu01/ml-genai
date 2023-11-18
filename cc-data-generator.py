import pandas as pd
import random
from datetime import datetime, timedelta

# Number of transactions to generate
num_transactions = 5000

# Sample attributes for transactions
transaction_types = ["CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN", "DEBIT"]
names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve"]
credit_card_numbers = [f"1234-5678-9012-{str(i).zfill(4)}" for i in range(1000)]

# Generate transactions
transactions = []
recent_transactions = {}

for _ in range(num_transactions):
    transaction_type = random.choice(transaction_types)
    name_orig = random.choice(names)
    name_dest = random.choice(names)
    while name_dest == name_orig:
        name_dest = random.choice(names)
    amount = round(random.uniform(10, 10000), 2)
    oldbalance_org = round(random.uniform(0, 50000), 2)
    newbalance_org = oldbalance_org - amount
    oldbalance_dest = round(random.uniform(0, 50000), 2)
    newbalance_dest = oldbalance_dest + amount
    credit_card = random.choice(credit_card_numbers)
    timestamp = datetime.now() - timedelta(days=random.randint(0, 365))
    
    # Simulate fraud based on patterns
    is_fraud = 0
    if amount > 9000:  # Large transactions
        is_fraud = 1
    elif transaction_type == "TRANSFER" and amount > 5000:  # Uncommon transaction type with large amount
        is_fraud = 1
    elif credit_card in recent_transactions and (timestamp - recent_transactions[credit_card]).seconds < 60:  # Frequent transactions in short time
        is_fraud = 1
    
    # Update recent transactions
    recent_transactions[credit_card] = timestamp
    
    transactions.append([transaction_type, name_orig, name_dest, amount, oldbalance_org, newbalance_org, oldbalance_dest, newbalance_dest, is_fraud, credit_card, timestamp])

# Convert to DataFrame
df = pd.DataFrame(transactions, columns=["type", "nameOrig", "nameDest", "amount", "oldbalanceOrg", "newbalanceOrg", "oldbalanceDest", "newbalanceDest", "isFraud", "creditCard", "timestamp"])

# Save to CSV
df.to_csv("simulated_transactions.csv", index=False)

print("Simulated transactions saved to 'simulated_transactions.csv'")
