import os
from datetime import datetime
from collections import defaultdict

# ========================
# LOGGING UTILITIES
# ========================
def setup_logging(subject_id):
    """Setup log file for debug output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{subject_id}_log_{timestamp}.txt"

    # Find next available filename
    counter = 1
    final_log_filename = log_filename
    while os.path.exists(final_log_filename):
        base_name = log_filename.replace('.txt', '')
        final_log_filename = f"{base_name}_{counter}.txt"
        counter += 1

    return final_log_filename


def log_message(log_file, message):
    """Write message to log file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)  


def log_sub_block_summary(log_file, main_block_idx, sub_block_idx, results, block_condition=None):
    """Log summary statistics for a completed sub-block"""
    # Filter results for this sub-block
    sub_results = [r for r in results
                   if r['main_block'] == main_block_idx and r['sub_block'] == sub_block_idx]

    if not sub_results:
        log_message(log_file, "No results found for this sub-block")
        return

    # Count same/different/NA responses
    same_count = len([r for r in sub_results if r['top_same'] == 'SAME'])
    diff_count = len([r for r in sub_results if r['top_same'] == 'DIFF'])
    na_count = len([r for r in sub_results if r['top_same'] == 'NA'])

    # Count correct/incorrect responses (excluding NA)
    correct_count = len([r for r in sub_results if r['correct'] == 'CORRECT'])
    incorrect_count = len([r for r in sub_results if r['correct'] == 'INCORRECT'])
    na_correct_count = len([r for r in sub_results if r['correct'] == 'NA'])

    # Calculate accuracy (excluding NA responses)
    total_responses = correct_count + incorrect_count
    accuracy = (correct_count / total_responses * 100) if total_responses > 0 else 0

    # Count by category
    category_counts = defaultdict(
        lambda: {'SAME': 0, 'DIFF': 0, 'NA': 0, 'CORRECT': 0, 'INCORRECT': 0, 'NA_CORRECT': 0})
    for r in sub_results:
        category_counts[r['category']][r['top_same']] += 1
        category_counts[r['category']][r['correct']] += 1

    log_message(log_file, f"\n{'=' * 60}")
    header = f"SUB-BLOCK {sub_block_idx} SUMMARY (MAIN BLOCK {main_block_idx}"
    if block_condition:
        header += f" - {block_condition.upper()}"
    header += ")"
    log_message(log_file, header)
    log_message(log_file, f"{'=' * 60}")
    log_message(log_file, f"Total trials: {len(sub_results)}")
    log_message(log_file, f"")
    log_message(log_file, f"TOP HALF COMPARISON:")
    log_message(log_file, f"  SAME: {same_count}")
    log_message(log_file, f"  DIFFERENT: {diff_count}")
    log_message(log_file, f"  NA (first trials): {na_count}")
    if same_count + diff_count > 0:
        same_pct = same_count / (same_count + diff_count) * 100
        diff_pct = diff_count / (same_count + diff_count) * 100
        log_message(log_file, f"  Same/Different Balance: {same_count}/{diff_count} = {same_pct:.1f}%/{diff_pct:.1f}%")
    log_message(log_file, f"")
    log_message(log_file, f"RESPONSE ACCURACY:")
    log_message(log_file, f"  CORRECT: {correct_count}")
    log_message(log_file, f"  INCORRECT: {incorrect_count}")
    log_message(log_file, f"  NA (no response required): {na_correct_count}")
    if total_responses > 0:
        log_message(log_file, f"  Accuracy: {accuracy:.1f}% ({correct_count}/{total_responses})")
    log_message(log_file, f"")
    log_message(log_file, f"BY CATEGORY:")
    for category, counts in category_counts.items():
        log_message(log_file, f"  {category.upper()}:")
        log_message(log_file, f"    Same: {counts['SAME']}, Different: {counts['DIFF']}, NA: {counts['NA']}")
        cat_correct = counts['CORRECT']
        cat_incorrect = counts['INCORRECT']
        cat_total = cat_correct + cat_incorrect
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        log_message(log_file, f"    Accuracy: {cat_accuracy:.1f}% ({cat_correct}/{cat_total})")
    log_message(log_file, f"{'=' * 60}\n")


def log_main_block_summary(log_file, main_block_idx, results, block_condition=None):
    """Log summary statistics for a completed main block"""
    # Filter results for this main block
    main_results = [r for r in results if r['main_block'] == main_block_idx]

    if not main_results:
        log_message(log_file, "No results found for this main block")
        return

    # Count same/different/NA responses
    same_count = len([r for r in main_results if r['top_same'] == 'SAME'])
    diff_count = len([r for r in main_results if r['top_same'] == 'DIFF'])
    na_count = len([r for r in main_results if r['top_same'] == 'NA'])

    # Count correct/incorrect responses (excluding NA)
    correct_count = len([r for r in main_results if r['correct'] == 'CORRECT'])
    incorrect_count = len([r for r in main_results if r['correct'] == 'INCORRECT'])
    na_correct_count = len([r for r in main_results if r['correct'] == 'NA'])

    # Calculate accuracy (excluding NA responses)
    total_responses = correct_count + incorrect_count
    accuracy = (correct_count / total_responses * 100) if total_responses > 0 else 0

    # Count by category
    category_counts = defaultdict(
        lambda: {'SAME': 0, 'DIFF': 0, 'NA': 0, 'CORRECT': 0, 'INCORRECT': 0, 'NA_CORRECT': 0})
    for r in main_results:
        category_counts[r['category']][r['top_same']] += 1
        category_counts[r['category']][r['correct']] += 1

    # Count by sub-block
    sub_block_counts = defaultdict(
        lambda: {'trials': 0, 'same': 0, 'diff': 0, 'na': 0, 'correct': 0, 'incorrect': 0})
    for r in main_results:
        sub_block_counts[r['sub_block']]['trials'] += 1
        if r['top_same'] == 'SAME':
            sub_block_counts[r['sub_block']]['same'] += 1
        elif r['top_same'] == 'DIFF':
            sub_block_counts[r['sub_block']]['diff'] += 1
        elif r['top_same'] == 'NA':
            sub_block_counts[r['sub_block']]['na'] += 1

        if r['correct'] == 'CORRECT':
            sub_block_counts[r['sub_block']]['correct'] += 1
        elif r['correct'] == 'INCORRECT':
            sub_block_counts[r['sub_block']]['incorrect'] += 1

    log_message(log_file, f"\n{'=' * 60}")
    # Add block condition to header
    header = f"MAIN BLOCK {main_block_idx} SUMMARY"
    if block_condition:
        header += f" - {block_condition.upper()}"
    log_message(log_file, header)
    log_message(log_file, f"{'=' * 60}")
    log_message(log_file, f"Total trials: {len(main_results)}")
    log_message(log_file, f"Total sub-blocks: {len(sub_block_counts)}")
    log_message(log_file, f"")
    log_message(log_file, f"OVERALL TOP HALF COMPARISON:")
    log_message(log_file, f"  SAME: {same_count}")
    log_message(log_file, f"  DIFFERENT: {diff_count}")
    log_message(log_file, f"  NA (first trials): {na_count}")
    if same_count + diff_count > 0:
        same_pct = same_count / (same_count + diff_count) * 100
        diff_pct = diff_count / (same_count + diff_count) * 100
        log_message(log_file, f"  Same/Different Balance: {same_count}/{diff_count} = {same_pct:.1f}%/{diff_pct:.1f}%")
    log_message(log_file, f"")
    log_message(log_file, f"OVERALL RESPONSE ACCURACY:")
    log_message(log_file, f"  CORRECT: {correct_count}")
    log_message(log_file, f"  INCORRECT: {incorrect_count}")
    log_message(log_file, f"  NA (no response required): {na_correct_count}")
    if total_responses > 0:
        log_message(log_file, f"  Accuracy: {accuracy:.1f}% ({correct_count}/{total_responses})")
    log_message(log_file, f"")

    # Category breakdown for main block
    log_message(log_file, f"BY CATEGORY:")
    for category, counts in category_counts.items():
        log_message(log_file, f"  {category.upper()}:")
        log_message(log_file, f"    Same: {counts['SAME']}, Different: {counts['DIFF']}, NA: {counts['NA']}")
        cat_correct = counts['CORRECT']
        cat_incorrect = counts['INCORRECT']
        cat_total = cat_correct + cat_incorrect
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        log_message(log_file, f"    Accuracy: {cat_accuracy:.1f}% ({cat_correct}/{cat_total})")
    log_message(log_file, f"")

    log_message(log_file, f"BY SUB-BLOCK:")
    for sub_block_num in sorted(sub_block_counts.keys()):
        counts = sub_block_counts[sub_block_num]
        sub_accuracy = (counts['correct'] / (counts['correct'] + counts['incorrect']) * 100) if (counts['correct'] +
                                                                                                 counts[
                                                                                                     'incorrect']) > 0 else 0
        log_message(log_file, f"  Sub-block {sub_block_num}: {counts['trials']} trials")
        log_message(log_file, f"    Same: {counts['same']}, Different: {counts['diff']}, NA: {counts['na']}")
        log_message(log_file,
                    f"    Accuracy: {sub_accuracy:.1f}% ({counts['correct']}/{counts['correct'] + counts['incorrect']})")
    log_message(log_file, f"{'=' * 60}\n")


def log_experiment_summary(log_file, results, subject_id):
    """Log overall experiment summary with detailed category-by-condition breakdown"""
    total_trials = len(results)
    total_same = len([r for r in results if r['top_same'] == 'SAME'])
    total_diff = len([r for r in results if r['top_same'] == 'DIFF'])
    total_na = len([r for r in results if r['top_same'] == 'NA'])

    total_correct = len([r for r in results if r['correct'] == 'CORRECT'])
    total_incorrect = len([r for r in results if r['correct'] == 'INCORRECT'])
    total_responses = total_correct + total_incorrect
    overall_accuracy = (total_correct / total_responses * 100) if total_responses > 0 else 0

    # Count by main_block
    block_counts = defaultdict(
        lambda: {'trials': 0, 'same': 0, 'diff': 0, 'na': 0, 'correct': 0, 'incorrect': 0, 'condition': ''})
    for r in results:
        block_num = r['main_block']
        block_counts[block_num]['trials'] += 1
        block_counts[block_num]['condition'] = r['block_condition']  # Store the condition name
        if r['top_same'] == 'SAME':
            block_counts[block_num]['same'] += 1
        elif r['top_same'] == 'DIFF':
            block_counts[block_num]['diff'] += 1
        elif r['top_same'] == 'NA':
            block_counts[block_num]['na'] += 1

        if r['correct'] == 'CORRECT':
            block_counts[block_num]['correct'] += 1
        elif r['correct'] == 'INCORRECT':
            block_counts[block_num]['incorrect'] += 1

    # Count by category across all conditions
    category_counts = defaultdict(
        lambda: {'SAME': 0, 'DIFF': 0, 'NA': 0, 'CORRECT': 0, 'INCORRECT': 0, 'trials': 0})
    for r in results:
        cat = r['category']
        category_counts[cat]['trials'] += 1
        category_counts[cat][r['top_same']] += 1
        category_counts[cat][r['correct']] += 1

    # Count by condition (block_condition)
    condition_counts = defaultdict(
        lambda: {'trials': 0, 'same': 0, 'diff': 0, 'na': 0, 'correct': 0, 'incorrect': 0})
    for r in results:
        condition = r['block_condition']
        condition_counts[condition]['trials'] += 1
        if r['top_same'] == 'SAME':
            condition_counts[condition]['same'] += 1
        elif r['top_same'] == 'DIFF':
            condition_counts[condition]['diff'] += 1
        elif r['top_same'] == 'NA':
            condition_counts[condition]['na'] += 1

        if r['correct'] == 'CORRECT':
            condition_counts[condition]['correct'] += 1
        elif r['correct'] == 'INCORRECT':
            condition_counts[condition]['incorrect'] += 1

    # Count by category within each condition 
    condition_category_counts = defaultdict(lambda: defaultdict(
        lambda: {'SAME': 0, 'DIFF': 0, 'NA': 0, 'CORRECT': 0, 'INCORRECT': 0, 'trials': 0}))
    for r in results:
        condition = r['block_condition']
        category = r['category']
        condition_category_counts[condition][category]['trials'] += 1
        condition_category_counts[condition][category][r['top_same']] += 1
        condition_category_counts[condition][category][r['correct']] += 1

    log_message(log_file, f"\n{'=' * 80}")
    log_message(log_file, f"EXPERIMENT SUMMARY FOR SUBJECT {subject_id}")
    log_message(log_file, f"{'=' * 80}")
    log_message(log_file, f"Total trials across all blocks: {total_trials}")
    log_message(log_file, f"Total main blocks: {len(block_counts)}")
    log_message(log_file, f"Total sub-blocks: {len(set((r['main_block'], r['sub_block']) for r in results))}")
    log_message(log_file, f"")
    log_message(log_file, f"OVERALL TOP HALF COMPARISONS:")
    log_message(log_file, f"  SAME: {total_same}")
    log_message(log_file, f"  DIFFERENT: {total_diff}")
    log_message(log_file, f"  NA (first trials): {total_na}")
    if total_same + total_diff > 0:
        same_pct = total_same / (total_same + total_diff) * 100
        diff_pct = total_diff / (total_same + total_diff) * 100
        log_message(log_file, f"  Overall Same/Different Balance: {same_pct:.1f}%/{diff_pct:.1f}%")
    log_message(log_file, f"")
    log_message(log_file, f"OVERALL RESPONSE ACCURACY:")
    log_message(log_file, f"  CORRECT: {total_correct}")
    log_message(log_file, f"  INCORRECT: {total_incorrect}")
    if total_responses > 0:
        log_message(log_file, f"  Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_responses})")
    log_message(log_file, f"")

    # Enhanced block breakdown
    log_message(log_file, f"BY MAIN BLOCK:")
    for block_num in sorted(block_counts.keys()):
        counts = block_counts[block_num]
        block_accuracy = (counts['correct'] / (counts['correct'] + counts['incorrect']) * 100) if (counts['correct'] +
                                                                                                   counts[
                                                                                                       'incorrect']) > 0 else 0
        log_message(log_file, f"  Block {block_num} ({counts['condition']}): {counts['trials']} trials")
        log_message(log_file, f"    Same: {counts['same']}, Different: {counts['diff']}, NA: {counts['na']}")
        log_message(log_file,
                    f"    Accuracy: {block_accuracy:.1f}% ({counts['correct']}/{counts['correct'] + counts['incorrect']})")
    log_message(log_file, f"")

    # Overall category breakdown
    log_message(log_file, f"BY CATEGORY (ACROSS ALL CONDITIONS):")
    for category in sorted(category_counts.keys()):
        counts = category_counts[category]
        cat_correct = counts['CORRECT']
        cat_incorrect = counts['INCORRECT']
        cat_total = cat_correct + cat_incorrect
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        log_message(log_file, f"  {category.upper()}: {counts['trials']} trials")
        log_message(log_file, f"    Same: {counts['SAME']}, Different: {counts['DIFF']}, NA: {counts['NA']}")
        log_message(log_file, f"    Accuracy: {cat_accuracy:.1f}% ({cat_correct}/{cat_total})")
    log_message(log_file, f"")

    # Condition breakdown
    log_message(log_file, f"BY CONDITION:")
    for condition in sorted(condition_counts.keys()):
        counts = condition_counts[condition]
        cond_accuracy = (counts['correct'] / (counts['correct'] + counts['incorrect']) * 100) if (counts['correct'] +
                                                                                                  counts[
                                                                                                      'incorrect']) > 0 else 0
        log_message(log_file, f"  {condition.upper()}: {counts['trials']} trials")
        log_message(log_file, f"    Same: {counts['same']}, Different: {counts['diff']}, NA: {counts['na']}")
        log_message(log_file,
                    f"    Accuracy: {cond_accuracy:.1f}% ({counts['correct']}/{counts['correct'] + counts['incorrect']})")
    log_message(log_file, f"")

    # Category breakdown within each condition
    log_message(log_file, f"DETAILED ANALYSIS: BY CATEGORY WITHIN EACH CONDITION")
    log_message(log_file, f"=" * 60)
    for condition in sorted(condition_category_counts.keys()):
        log_message(log_file, f"\n{condition.upper()} CONDITION:")
        log_message(log_file, f"-" * 40)

        # Calculate human face accuracy in this condition
        human_categories = [cat for cat in condition_category_counts[condition].keys() if 'human' in cat]
        monkey_categories = [cat for cat in condition_category_counts[condition].keys() if 'monkey' in cat]

        for category in sorted(condition_category_counts[condition].keys()):
            counts = condition_category_counts[condition][category]
            cat_correct = counts['CORRECT']
            cat_incorrect = counts['INCORRECT']
            cat_total = cat_correct + cat_incorrect
            cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0

            log_message(log_file, f"  {category.upper()}: {counts['trials']} trials")
            log_message(log_file, f"    Same: {counts['SAME']}, Different: {counts['DIFF']}, NA: {counts['NA']}")
            log_message(log_file, f"    Accuracy: {cat_accuracy:.1f}% ({cat_correct}/{cat_total})")

        # Summary for this condition
        if human_categories and monkey_categories:
            human_cat = human_categories[0]  
            monkey_cat = monkey_categories[0]  

            human_acc = condition_category_counts[condition][human_cat]
            human_correct = human_acc['CORRECT']
            human_total = human_correct + human_acc['INCORRECT']
            human_accuracy = (human_correct / human_total * 100) if human_total > 0 else 0

            monkey_acc = condition_category_counts[condition][monkey_cat]
            monkey_correct = monkey_acc['CORRECT']
            monkey_total = monkey_correct + monkey_acc['INCORRECT']
            monkey_accuracy = (monkey_correct / monkey_total * 100) if monkey_total > 0 else 0

            log_message(log_file, f"  CONDITION SUMMARY:")
            log_message(log_file, f"    Human accuracy: {human_accuracy:.1f}% ({human_correct}/{human_total})")
            log_message(log_file, f"    Monkey accuracy: {monkey_accuracy:.1f}% ({monkey_correct}/{monkey_total})")
            log_message(log_file, f"    Difference: {human_accuracy - monkey_accuracy:.1f}% (Human - Monkey)")

    # Hypothesis-focused analysis
    log_message(log_file, f"\n{'=' * 60}")
    log_message(log_file, f"HYPOTHESIS ANALYSIS: ALIGNED vs MISALIGNED MONKEY EFFECTS")
    log_message(log_file, f"{'=' * 60}")

    # Compare human face accuracy across aligned vs misaligned conditions
    aligned_conditions = [cond for cond in condition_category_counts.keys() if
                          'aligned' in cond and 'misaligned' not in cond]
    misaligned_conditions = [cond for cond in condition_category_counts.keys() if 'misaligned' in cond]

    if aligned_conditions and misaligned_conditions:
        log_message(log_file, f"\nHUMAN FACE ACCURACY COMPARISON:")
        log_message(log_file, f"-" * 40)

        # Calculate human accuracy in aligned conditions
        aligned_human_correct = 0
        aligned_human_total = 0
        for condition in aligned_conditions:
            human_cats = [cat for cat in condition_category_counts[condition].keys() if 'human' in cat]
            for human_cat in human_cats:
                counts = condition_category_counts[condition][human_cat]
                aligned_human_correct += counts['CORRECT']
                aligned_human_total += counts['CORRECT'] + counts['INCORRECT']

        aligned_human_accuracy = (aligned_human_correct / aligned_human_total * 100) if aligned_human_total > 0 else 0

        # Calculate human accuracy in misaligned conditions
        misaligned_human_correct = 0
        misaligned_human_total = 0
        for condition in misaligned_conditions:
            human_cats = [cat for cat in condition_category_counts[condition].keys() if 'human' in cat]
            for human_cat in human_cats:
                counts = condition_category_counts[condition][human_cat]
                misaligned_human_correct += counts['CORRECT']
                misaligned_human_total += counts['CORRECT'] + counts['INCORRECT']

        misaligned_human_accuracy = (
                    misaligned_human_correct / misaligned_human_total * 100) if misaligned_human_total > 0 else 0

        log_message(log_file,
                    f"Human faces with ALIGNED monkeys: {aligned_human_accuracy:.1f}% ({aligned_human_correct}/{aligned_human_total})")
        log_message(log_file,
                    f"Human faces with MISALIGNED monkeys: {misaligned_human_accuracy:.1f}% ({misaligned_human_correct}/{misaligned_human_total})")
        log_message(log_file,
                    f"Difference: {misaligned_human_accuracy - aligned_human_accuracy:.1f}% (Misaligned - Aligned)")

        if aligned_human_accuracy > misaligned_human_accuracy:
            log_message(log_file,
                        f"RESULT: Human face processing was BETTER with aligned monkeys")
        elif misaligned_human_accuracy > aligned_human_accuracy:
            log_message(log_file,
                        f"RESULT: Human face processing was BETTER with misaligned monkeys")
        else:
            log_message(log_file, f"RESULT: No difference between conditions")

    log_message(log_file, f"{'=' * 80}\n")