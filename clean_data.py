import csv
import io
import re

# Read file as raw bytes to handle BOM
with open('datasets/xauusd_2016-2025.csv', 'rb') as f:
    raw = f.read()

# Strip BOM
for bom in [b'\xef\xbb\xbf', b'\xff\xfe', b'\xfe\xff']:
    raw = raw.lstrip(bom)

content = raw.decode('utf-8')
# Also strip any unicode BOM char
content = content.lstrip('\ufeff')

reader = csv.reader(io.StringIO(content))
rows = list(reader)
header = rows[0]

# Strip stray quotes and whitespace from header
header = [re.sub(r'["\ufeff]', '', h).strip() for h in header]
print('Original header:', header)

# Find and remove Vol. column
vol_idx = None
for i, h in enumerate(header):
    if 'vol' in h.lower():
        vol_idx = i
        break

print(f'Vol. column index: {vol_idx}')

# Build clean data: lowercase headers, remove Vol., remove commas from numbers
new_header = [h.lower() for i, h in enumerate(header) if i != vol_idx]
clean_rows = [new_header]

for row in rows[1:]:
    new_row = []
    for i, cell in enumerate(row):
        if i == vol_idx:
            continue
        # Remove commas (thousand separators) and stray quotes
        cleaned = cell.replace(',', '').strip('"').strip()
        new_row.append(cleaned)
    clean_rows.append(new_row)

# Write back without BOM
with open('datasets/xauusd_2016-2025.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(clean_rows)

# Verify
for r in clean_rows[:6]:
    print(r)
print(f'Total rows (incl header): {len(clean_rows)}')
