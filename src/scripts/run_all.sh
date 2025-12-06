#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Global variables to track state
DATA_LOADED=0
CURRENT_IMPL=""
RESULTS_FILE="results.txt"
SELECTED_FILE=""


> "$RESULTS_FILE"

# Function to validate menu integers
validate_number() {
    if [[ ! "$1" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid input. Please enter a number.${NC}"
        return 1
    fi
    return 0
}

# Function to validate algorithm parameters
validate_param() {
    local input=$1
    # If input is empty, use default
    if [ -z "$input" ]; then
        return 0
    fi

    # Check for valid number
    if [[ ! "$input" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
        echo -e "${RED}Error: '$input' is not a valid numeric value.${NC}"
        return 1
    fi
    return 0
}

# Function to get current timestamp
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Cleaner
create_cleaner_script() {
    cat << 'EOF' > clean_data.py
import argparse
import numpy as np
import pandas as pd
import sys

# Setup argument parser
parser = argparse.ArgumentParser(description="Clean CSV file")
parser.add_argument("--input", type=str, required=True, help="Input CSV file")
parser.add_argument("--output", type=str, required=True, help="Output CSV file")

args = parser.parse_args()

input_file = args.input
output_file = args.output

try:
    income_df = pd.read_csv(input_file)

    if income_df.columns[0] == '' or 'Unnamed' in str(income_df.columns[0]):
        income_df = income_df.iloc[:, 1:]

    if len(income_df.columns) != 14:
        raise ValueError(f"Expected 14 columns, but found {len(income_df.columns)} columns")

    # Check if this looks like the adult income dataset before proceeding
    required_cols = ['age', 'workclass', 'education', 'occupation', 'income']
    if not all(col in income_df.columns for col in required_cols):
        print("Dataset does not appear to be Adult Income data. Skipping specific cleaning.")
        sys.exit(1)

    cleaned_rows = []
    seen = set()

    # Cast to string where appropriate
    str_cols = ['workclass', 'education', 'marital.status', 'occupation',
                'relationship', 'race', 'sex', 'native.country', 'income']

    for col in str_cols:
        if col in income_df.columns:
            income_df[col] = income_df[col].astype('string')

    # Standard cleaning
    income_df.replace('?', np.nan, inplace=True)
    income_df.dropna(how='any', inplace=True)
    income_df.drop_duplicates(inplace=True)

    # Validation Dictionary
    valid_values = {
        'workclass': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
        'education': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'],
        'marital.status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
        'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'],
        'relationship': ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'],
        'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
        'sex': ['Female', 'Male'],
        'native.country': ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'],
        'income': ['<=50K', '>50K']
    }

    # Check for invalid numerical values
    numerical_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    for col in numerical_columns:
        if col in income_df.columns:
            try:
                pd.to_numeric(income_df[col], errors='raise')
            except (ValueError, TypeError) as e:
                raise ValueError(f"Column '{col}' contains non-numeric values") from None

    # Check for invalid categorical values
    for col, valid_list in valid_values.items():
        if col in income_df.columns:
            # Filter rows that exist but are not in valid list
            invalid_mask = ~income_df[col].isin(valid_list)
            if invalid_mask.any():
                print(f"Dropping {invalid_mask.sum()} rows with invalid values in {col}")
                income_df = income_df[~invalid_mask]

    income_df.to_csv(output_file, index=False)
    print(f"Successfully cleaned data. Saved to {output_file}")

except Exception as e:
    print(f"Python Cleaning Error: {e}")
    sys.exit(1)
EOF
}

# Function to scan data folder and populate global array
declare -a DATASETS
scan_data_folder() {
    echo -e "\n${BLUE}Available datasets in data folder:${NC}"
    local i=1
    DATASETS=()

    if [ -d "../data" ]; then
        for file in ../data/*.csv; do
            if [ -f "$file" ]; then
                DATASETS+=("$(basename "$file")")
                echo "  $i. $(basename "$file")"
                ((i++))
            fi
        done
    fi

    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo -e "${RED}No CSV files found in data folder.${NC}"
        return 1
    fi

    return 0
}

# Function to load data
load_data() {
    echo -e "\n${YELLOW}Loading and cleaning input data set:${NC}"
    echo "************************************"

    if ! scan_data_folder; then
        return 1
    fi

    echo ""
    read -p "Enter dataset number: " choice

    if ! validate_number "$choice"; then
        return 1
    fi

    if [ "$choice" -lt 1 ] || [ "$choice" -gt ${#DATASETS[@]} ]; then
        echo -e "${RED}Error: Invalid dataset number.${NC}"
        return 1
    fi

    local RAW_FILE="${DATASETS[$((choice-1))]}"

    if command -v python3 &> /dev/null; then
        # Generate the Python script
        create_cleaner_script

        if [[ "$RAW_FILE" == *"_cleaned.csv" ]]; then
            SELECTED_FILE="$RAW_FILE"
            echo -e "${GREEN}File appears to be pre-cleaned. Using: $SELECTED_FILE${NC}"
        else
            local BASE_NAME="${RAW_FILE%.*}"
            local CLEAN_NAME="${BASE_NAME}_cleaned.csv"

            echo -e "${YELLOW}Attempting to run Python pre-processing on $RAW_FILE...${NC}"

            python3 clean_data.py --input "../data/$RAW_FILE" --output "../data/$CLEAN_NAME"

            if [ $? -eq 0 ]; then
                SELECTED_FILE="$CLEAN_NAME"
                echo -e "${GREEN}Data cleaned successfully! Using: $SELECTED_FILE${NC}"
            else
                echo -e "${RED}ERROR: Data cleaning failed.${NC}"
                echo -e "${RED}Cannot proceed with file: $RAW_FILE${NC}"
                rm -f clean_data.py
                DATA_LOADED=0
                return 1
            fi

            rm -f clean_data.py
        fi
    else
        SELECTED_FILE="$RAW_FILE"
        echo -e "${YELLOW}Python3 not found. Skipping cleaning step. Using: $SELECTED_FILE${NC}"
    fi

    echo -e "\n${GREEN}[$(timestamp)] Starting Script${NC}"
    echo -e "${GREEN}[$(timestamp)] Selected data set: $SELECTED_FILE${NC}"

    # Get file info
    if [ -f "../data/$SELECTED_FILE" ]; then
        local cols=$(head -1 "../data/$SELECTED_FILE" | tr ',' '\n' | wc -l)
        local rows=$(($(wc -l < "../data/$SELECTED_FILE") - 1))

        echo -e "${GREEN}[$(timestamp)] Total Columns Read: $cols${NC}"
        echo -e "${GREEN}[$(timestamp)] Total Rows Read: $rows${NC}"

        DATA_LOADED=1
    else
        echo -e "${RED}Error: File not found.${NC}"
        DATA_LOADED=0
        return 1
    fi
}

# Function to run Linear Regression
run_linear_regression() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Linear Regression (closed-form):${NC}"
    echo "********************************"

    read -p "Target variable [hours.per.week]: " target
    target=${target:-hours.per.week}

    read -p "L2 regularization [0.1]: " l2
    if ! validate_param "$l2"; then return 1; fi
    l2=${l2:-0.1}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)

    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                echo "Compiling C++ implementation..."
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                --test "$DATA_PATH" \
                --target "$target" \
                --algo linear \
                --l2 "$l2" \
                --normalize 2>&1 | tee /tmp/output.txt
                            cd ../scripts
                            ;;
                        "Java")
                            cd ../oop-java
                            if [ ! -f "app/Main.class" ]; then
                                echo "Compiling Java implementation..."
                                javac $(find . -name "*.java")
                            fi
                            echo "2" | java app.Main --train "$DATA_PATH" \
                                --normalize \
                                --target "$target" \
                                --l2 "$l2" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                        "Lisp")
                                                            cd ../fp
                                                            sbcl --script main.lisp --algo linear --train "$DATA_PATH" --target "$target" --l2 "$l2" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                    esac

                                                    local end_time=$(date +%s.%N)
                                                    local elapsed=$(echo "$end_time - $start_time" | bc)

    # Extract metrics from output
    local rmse=$(grep -i "RMSE" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local r2=$(grep -i "R\^2\|R²" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    # Log the results only if run was successful
    if [ -z "$rmse" ] || [ -z "$r2" ]; then
        echo -e "${RED}Run failed.${NC}"
    else
        echo "$CURRENT_IMPL,Linear Regression,$elapsed,$rmse,$r2,$sloc" >> "$RESULTS_FILE"
    fi
}

# Function to run Logistic Regression
run_logistic_regression() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Logistic Regression (binary):${NC}"
    echo "*****************************"

    read -p "Target variable [income]: " target
    target=${target:-income}

    read -p "Learning rate [0.2]: " lr
    if ! validate_param "$lr"; then return 1; fi
    lr=${lr:-0.2}

    read -p "Epochs [400]: " epochs
    if ! validate_param "$epochs"; then return 1; fi
    epochs=${epochs:-400}

    read -p "L2 regularization [0.003]: " l2
    if ! validate_param "$l2"; then return 1; fi
    l2=${l2:-0.003}

    read -p "Random seed [7]: " seed
    if ! validate_param "$seed"; then return 1; fi
    seed=${seed:-7}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)

    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                echo "Compiling C++ implementation..."
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                --test "$DATA_PATH" \
                --target "$target" \
                --algo logistic \
                --lr "$lr" \
                --epochs "$epochs" \
                --l2 "$l2" \
                --normalize 2>&1 | tee /tmp/output.txt
                            cd ../scripts
                            ;;
                        "Java")
                            cd ../oop-java
                            echo "Compiling Java implementation..."
                            javac $(find . -name "*.java") > /dev/null 2>&1
                            printf "1\n3\n" | stdbuf -oL java app.Main --train "$DATA_PATH" --target "$target" --normalize --lr "$lr" --epochs "$epochs" --l2 "$l2" --seed "$seed" 2>&1 | tee /tmp/output.txt

                            cd ../scripts
                            ;;
                        "Lisp")
                            cd ../fp
                            sbcl --script main.lisp --algo logistic \
                                --train "$DATA_PATH" --target "$target" \
                                --lr "$lr" --epochs "$epochs" --l2 "$l2" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                    esac

                                                    local end_time=$(date +%s.%N)
                                                    local elapsed=$(echo "$end_time - $start_time" | bc)

    # Extract metrics
    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local f1=$(grep -i "Macro-F1\|F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    # Log results only if metrics exist
    if [ -z "$acc" ]; then
        echo -e "${RED}Run failed.${NC}"
    else
        echo "$CURRENT_IMPL,Logistic Regression,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
    fi
}

# Function to run k-Nearest Neighbors
run_knn() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}k-Nearest Neighbors:${NC}"
    echo "********************"

    read -p "Number of neighbors (k) [5]: " k
    if ! validate_param "$k"; then return 1; fi
    k=${k:-5}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)
    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                --test "$DATA_PATH" \
                --target "$target" \
                --algo knn \
                --k "$k" \
                --normalize 2>&1 | tee /tmp/output.txt
                            cd ../scripts
                            ;;
                        "Java")
                            cd ../oop-java
                            if [ ! -f "app/Main.class" ]; then
                                javac $(find . -name "*.java") > /dev/null 2>&1
                            fi
                            echo "4" | java app.Main --train "$DATA_PATH" --target "$target" \
                                --normalize \
                                --k "$k" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                        "Lisp")
                                                            cd ../fp
                                                            sbcl --script main.lisp --algo knn --target "$target" --train "$DATA_PATH" --k "$k" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                    esac

                                                    local end_time=$(date +%s.%N)
                                                    local elapsed=$(echo "$end_time - $start_time" | bc)

                                                    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
                                                    local f1=$(grep -i "Macro-F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
                                                    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

                                                    if [ -z "$acc" ]; then
                                                        echo -e "${RED}Run failed.${NC}"
                                                    else
                                                        echo "$CURRENT_IMPL,k-Nearest Neighbors,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
                                                    fi
                                                }

# Function to run Decision Tree
run_decision_tree() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Decision Tree (ID3):${NC}"
    echo "********************"

    read -p "Max depth [5]: " depth
    if ! validate_param "$depth"; then return 1; fi
    depth=${depth:-5}

    read -p "Number of bins [10]: " bins
    if ! validate_param "$bins"; then return 1; fi
    bins=${bins:-10}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)
    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                --test "$DATA_PATH" \
                --target "$target" \
                --algo tree \
                --max_depth "$depth" \
                --normalize 2>&1 | tee /tmp/output.txt
                            cd ../scripts
                            ;;
                        "Java")
                            cd ../oop-java
                            if [ ! -f "app/Main.class" ]; then
                                javac $(find . -name "*.java") > /dev/null 2>&1
                            fi
                            echo "5" | java app.Main --train "$DATA_PATH" --target "$target" \
                                --normalize \
                                --max_depth "$depth" \
                                --bins "$bins" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                        "Lisp")
                                                            cd ../fp
                                                            sbcl --script main.lisp --algo tree \
                                                                --train "$DATA_PATH" --target "$target" \
                                                                --max_depth "$depth" --n_bins "$bins" 2>&1 | tee /tmp/output.txt
                                                                                                                            cd ../scripts
                                                                                                                            ;;
                                                                                                                    esac

                                                                                                                    local end_time=$(date +%s.%N)
                                                                                                                    local elapsed=$(echo "$end_time - $start_time" | bc)

                                                                                                                    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
                                                                                                                    local f1=$(grep -i "Macro-F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
                                                                                                                    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

                                                                                                                    if [ -z "$acc" ]; then
                                                                                                                        echo -e "${RED}Run failed.${NC}"
                                                                                                                    else
                                                                                                                        echo "$CURRENT_IMPL,Decision Tree,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
                                                                                                                    fi
                                                                                                                }

# Function to run Gaussian Naive Bayes
run_naive_bayes() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Gaussian Naive Bayes:${NC}"
    echo "*********************"

    read -p "Variance smoothing [1e-9]: " smooth
    if ! validate_param "$smooth"; then return 1; fi
    smooth=${smooth:-1e-9}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)
    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                --test "$DATA_PATH" \
                --target income \
                --algo nb \
                --normalize 2>&1 | tee /tmp/output.txt
                            cd ../scripts
                            ;;
                        "Java")
                            cd ../oop-java
                            if [ ! -f "app/Main.class" ]; then
                                javac $(find . -name "*.java") > /dev/null 2>&1
                            fi
                            echo "6" | java app.Main --train "$DATA_PATH" \
                                --normalize \
                                --smoothing "$smooth" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                        "Lisp")
                                                            cd ../fp
                                                            sbcl --script main.lisp --algo nb --train "$DATA_PATH" --smoothing "$smooth" 2>&1 | tee /tmp/output.txt
                                                            cd ../scripts
                                                            ;;
                                                    esac

                                                    local end_time=$(date +%s.%N)
                                                    local elapsed=$(echo "$end_time - $start_time" | bc)

                                                    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
                                                    local f1=$(grep -i "Macro-F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
                                                    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

                                                    if [ -z "$acc" ]; then
                                                        echo -e "${RED}Run failed.${NC}"
                                                    else
                                                        echo "$CURRENT_IMPL,Gaussian Naive Bayes,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
                                                    fi
                                                }

# Function to print implementation results
print_impl_results() {
    echo -e "\n${YELLOW}$CURRENT_IMPL Results:${NC}"
    echo "******************************"

    printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
        "Impl" "Algorithm" "TrainTime" "TestMetric1" "TestMetric2" "SLOC"
            echo "-----------------------------------------------------------------------------------------"

            if [ -f "$RESULTS_FILE" ]; then
                while IFS=',' read -r impl algo time m1 m2 sloc; do
                    if [ "$impl" == "$CURRENT_IMPL" ]; then
                        printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
                            "$impl" "$algo" "${time}s" "$m1" "$m2" "$sloc"
                    fi
                done < "$RESULTS_FILE"
            else
                echo "No results available yet."
            fi
        }

# Function to print general comparison results
print_general_results() {
    echo -e "\n${YELLOW}General Results (Comparison):${NC}"
    echo "*****************************"

    printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
        "Impl" "Algorithm" "TrainTime" "TestMetric1" "TestMetric2" "SLOC"
            echo "-----------------------------------------------------------------------------------------"

            if [ -f "$RESULTS_FILE" ] && [ -s "$RESULTS_FILE" ]; then
                while IFS=',' read -r impl algo time m1 m2 sloc; do
                    printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
                        "$impl" "$algo" "${time}s" "$m1" "$m2" "$sloc"
                                        done < "$RESULTS_FILE"
                                    else
                                        echo "No results available yet. Please run some algorithms first."
            fi
        }

# Algorithm menu
algorithm_menu() {
    while true; do
        echo -e "\n${BLUE}******************************************************${NC}"
        echo -e "${GREEN}You have selected $CURRENT_IMPL${NC}"
        echo -e "${BLUE}******************************************************${NC}"
        echo "Please select an option:"
        echo "(1) Load data"
        echo "(2) Linear Regression (closed-form)"
        echo "(3) Logistic Regression (binary)"
        echo "(4) k-Nearest Neighbors"
        echo "(5) Decision Tree (ID3)"
        echo "(6) Gaussian Naive Bayes"
        echo "(7) Print results"
        echo "(8) Back to main menu"
        echo ""

        read -p "Enter option: " choice

        if ! validate_number "$choice"; then
            continue
        fi

        case $choice in
            1) load_data ;;
            2) run_linear_regression ;;
            3) run_logistic_regression ;;
            4) run_knn ;;
            5) run_decision_tree ;;
            6) run_naive_bayes ;;
            7) print_impl_results ;;
            8) DATA_LOADED=0; break ;;
            *) echo -e "${RED}Invalid option. Please try again.${NC}" ;;
        esac
    done
}

# Main menu
main_menu() {
    while true; do
        echo -e "\n${BLUE}******************************************************${NC}"
        echo -e "${GREEN}Welcome to the AI/ML Library Implementation Comparison${NC}"
        echo -e "${BLUE}******************************************************${NC}"
        echo "Please select an implementation to run:"
        echo "(1) Procedural (C/C++)"
        echo "(2) Object-Oriented (Java)"
        echo "(3) Functional (Lisp)"
        echo "(4) Print General Results"
        echo "(5) Quit"
        echo ""

        read -p "Enter option: " choice

        if ! validate_number "$choice"; then
            continue
        fi

        case $choice in
            1)
                CURRENT_IMPL="C"
                algorithm_menu
                ;;
            2)
                CURRENT_IMPL="Java"
                algorithm_menu
                ;;
            3)
                CURRENT_IMPL="Lisp"
                algorithm_menu
                ;;
            4)
                print_general_results
                ;;
            5)
                echo -e "\n${GREEN}Thank you for using the comparison tool!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please enter a number between 1 and 5.${NC}"
                ;;
        esac
    done
}

# Check for required commands
check_dependencies() {
    local missing=0

    if ! command -v bc &> /dev/null; then
        echo -e "${RED}Error: 'bc' is required but not installed.${NC}"
        missing=1
    fi

    if ! command -v sbcl &> /dev/null && [ "$1" == "3" ]; then
        echo -e "${YELLOW}Warning: 'sbcl' not found. Lisp implementation will not work.${NC}"
    fi

    if ! command -v javac &> /dev/null && [ "$1" == "2" ]; then
        echo -e "${YELLOW}Warning: 'javac' not found. Java implementation will not work.${NC}"
    fi

    if ! command -v g++ &> /dev/null && [ "$1" == "1" ]; then
        echo -e "${YELLOW}Warning: 'g++' not found. C++ implementation will not work.${NC}"
    fi

    return $missing
}

# Start the script
check_dependencies
main_menu
