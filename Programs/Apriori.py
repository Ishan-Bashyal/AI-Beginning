from itertools import combinations

# Step 1: Transactions
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Diaper', 'Milk', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Diaper', 'Beer'],
    ['Bread', 'Butter', 'Cola'],
    ['Milk', 'Diaper', 'Bread', 'Butter'],
    ['Beer', 'Cola'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Diaper', 'Cola'],
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Diaper', 'Milk'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Butter', 'Cola'],
    ['Milk', 'Bread', 'Diaper', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Diaper', 'Bread'],
    ['Beer', 'Cola'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Diaper', 'Cola'],
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Diaper', 'Milk'],
    ['Milk', 'Diaper', 'Beer'],
    ['Bread', 'Butter', 'Cola'],
    ['Milk', 'Bread', 'Diaper', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Diaper', 'Bread', 'Butter'],
    ['Beer', 'Cola'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Diaper', 'Cola'],
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Diaper', 'Milk'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Butter', 'Cola'],
    ['Milk', 'Bread', 'Diaper', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Diaper', 'Bread'],
    ['Beer', 'Cola'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Diaper', 'Cola'],
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Diaper', 'Milk'],
    ['Milk', 'Diaper', 'Beer'],
    ['Bread', 'Butter', 'Cola'],
    ['Milk', 'Bread', 'Diaper', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Diaper', 'Bread', 'Butter'],
    ['Beer', 'Cola'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Diaper', 'Cola']
]


# Step 2: Count helper
def count_occurrences(itemset, transactions):
    return sum(1 for t in transactions if all(i in t for i in itemset))


# Step 3: Get unique items
unique_items = set(item for t in transactions for item in t)

# Step 4: Generate all frequent itemsets of size ≥ 2
min_support_count = 2  # Optional: filter out rare itemsets
rules = {}
total_transactions = len(transactions)

for size in range(2, 4):  # size 2 and 3 itemsets
    for itemset in combinations(sorted(unique_items), size):
        itemset = list(itemset)
        itemset_count = count_occurrences(itemset, transactions)
        if itemset_count < min_support_count:
            continue

        # Generate rules from itemset: A → B
        for i in range(1, len(itemset)):
            for lhs in combinations(itemset, i):
                rhs = set(itemset) - set(lhs)
                lhs = list(lhs)
                rhs = list(rhs)

                lhs_count = count_occurrences(lhs, transactions)
                if lhs_count == 0:
                    continue

                confidence = itemset_count / lhs_count
                if confidence >= 0.65:
                    rule_str = f"{' + '.join(lhs)} -> {' + '.join(rhs)}"
                    rules[rule_str] = confidence

# Step 5: Display results
print("✅ Strong Association Rules (Confidence ≥ 65%)\n")
for rule, conf in sorted(rules.items(), key=lambda x: -x[1]):
    print(f"Rule: {rule}, Confidence: {conf:.2%}")

