# TEST_1_Default_Pipeline

## Test 1: Simple Table (Baseline)

| Product   |   Q1 |   Q2 |   Q3 |
|-----------|------|------|------|
| Product A |  100 |  120 |  130 |
| Product B |   80 |   90 |   95 |

## Test 2: Table with Rowspan (Merged Rows)

| Category    | Product   | Price   |   Stock |
|-------------|-----------|---------|---------|
| Electronics | Laptop    | $999    |      50 |
| Electronics | Phone     | $699    |     120 |
| Electronics | Tablet    | $499    |      80 |
| Furniture   | Chair     | $199    |      30 |
| Furniture   | Desk      | $399    |      15 |

## Test 3: Table with Colspan (Merged Columns)

| Quarter   | Sales    | Sales         | Expenses   | Expenses   |
|-----------|----------|---------------|------------|------------|
|           | Domestic | International | Fixed      | Variable   |
| Q1        | $100K    | $50K          | $30K       | $20K       |
| Q2        | $120K    | $60K          | $30K       | $25K       |

## Test 4: Complex Table with Both Rowspan and Colspan

| Region   | Product   | 2023   | 2023   | 2023   | 2024   | 2024   | 2024   |
|----------|-----------|--------|--------|--------|--------|--------|--------|
| Region   | Product   | Q1     | Q2     | Q3     | Q1     | Q2     | Q3     |
| North    | Product A | 100    | 110    | 120    | 130    | 140    | 150    |
| North    | Product B | 80     | 85     | 90     | 95     | 100    | 105    |
| South    | Product A | 60     | 65     | 70     | 75     | 80     | 85     |
| South    | Product B | 50     | 52     | 54     | 56     | 58     | 60     |

## Test 5: Extremely Complex Table (Multi-level Headers + Merged Cells)

|      | Financial Metrics   | Financial Metrics   | Financial Metrics   | Financial Metrics   | Financial Metrics   | Financial Metrics   | Notes            |
|------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|
| Year | Revenue             | Revenue             | Revenue             | Profit              | Profit              | Profit              |                  |
|      | Actual              | Budget              | Variance            | Actual              | Budget              | Variance            |                  |
| 2022 | $1.2M               | $1.0M               | +20%                | $300K               | $250K               | +20%                | Strong growth    |
| 2023 | $1.5M               | $1.3M               | +15%                | $400K               | $350K               | +14%                | Market expansion |
| 2024 | $1.8M               | $1.6M               | +12%                | $500K               | $450K               | +11%                | Projected        |

## Test 6: Table with Empty Cells and Mixed Content

| Item          | Description       | Price   | Availability   |
|---------------|-------------------|---------|----------------|
| Laptop Bundle | Core i7, 16GB RAM | $999    | In Stock       |

| Item     | Description                    | Price                          | Availability   |
|----------|--------------------------------|--------------------------------|----------------|
|          | Includes: Mouse, Keyboard, Bag | Includes: Mouse, Keyboard, Bag | -              |
| Monitor  | 27" 4K Display                 | $499                           |                |
| Keyboard |                                | $79                            | Out of Stock   |