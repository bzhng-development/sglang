#!/bin/bash

# Create markdown file with header
cat > todo_blame_report.md << 'HEADER'
# TODO Blame Report for sglang

This report shows git blame information for all TODO/todo lines in the codebase.
Generated on $(date)

Total TODO lines found: $(rg --line-number --no-heading "TODO|todo" . | wc -l | tr -d ' ')
Total files with TODOs: $(rg --line-number --no-heading "TODO|todo" . | cut -d ':' -f1 | sort | uniq | wc -l | tr -d ' ')

---

HEADER

# Process each file with TODO lines
rg --line-number --no-heading "TODO|todo" . | while IFS=':' read -r file line_num rest; do
    # Skip if file doesn't exist or is binary
    if [[ ! -f "$file" ]] || file "$file" | grep -q "binary"; then
        continue
    fi
    
    echo "## $file" >> todo_blame_report.md
    echo "" >> todo_blame_report.md
    
    # Get git blame for the specific line
    blame_output=$(git blame -L "$line_num,$line_num" "$file" 2>/dev/null || echo "No blame info available")
    
    echo "\`\`\`" >> todo_blame_report.md
    echo "Line $line_num: $rest" >> todo_blame_report.md
    echo "Blame: $blame_output" >> todo_blame_report.md
    echo "\`\`\`" >> todo_blame_report.md
    echo "" >> todo_blame_report.md
done

echo "Report generated: todo_blame_report.md"
