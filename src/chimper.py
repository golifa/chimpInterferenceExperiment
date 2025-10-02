import os
import random
import re
import copy
import traceback
from psychopy import visual, event, core, gui, monitors
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime

from loggingUtils import setup_logging, log_message, log_sub_block_summary, log_main_block_summary, log_experiment_summary

# ========================
# EXPERIMENT SETTINGS
# ========================
BASE_PATH = r"CHANGE ME TO PATH" # Path to dataset folder
BLOCK_CONDITIONS = [
    ("female_faces/aligned", "monkey_faces/aligned"),
    ("female_faces/aligned", "monkey_faces/misaligned"),
    ("male_faces/aligned", "monkey_faces/aligned"),
    ("male_faces/aligned", "monkey_faces/misaligned")
]

SUB_BLOCKS_PER_MAIN_BLOCK = 3 # N of sub-blocks within each experimental condition
NUM_TRIALS_PER_SUB_BLOCK = 10  # Per category & per sub-block
STIMULUS_REPETITIONS = 4  # How many times each stimulus can be used in total
MAX_SEQUENCE_ATTEMPTS = 1000  # N of seed that it tries and grades based on usage_range, unused_count, consecutive_penalty
MAX_CONSECUTIVE_REPETITIONS = 3  # How many times each top half stimulus can be used consecutively
IMAGE_DISPLAY_TIME = 1 # Display time of stimulus
SCREEN_SIZE = (2560, 1440)
STIM_HEIGHT_PCT = 0.4 # Size of stimulus
INSTRUCTION_TEXT_COLOR = 'black'
INSTRUCTION_TEXT_SIZE = 40
BREAK_TEXT_COLOR = 'blue'
BREAK_TEXT_SIZE = 40
RESPONSE_TIME = 5 # Time given to respond after response window
WAIT_FIRST_TWO = 1 # Time to wait to skip the first two sets of stimulus in a sub-block


# ========================
# FILE PROCESSING
# ========================
def parse_filename(filename):
    match = re.search(r'top(\d+)bot(\d+)', filename, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def load_category(folder_path, category):
    trials = []
    if os.path.exists(folder_path):
        human_gender = None
        if "female" in folder_path:
            human_gender = "female"
        elif "male" in folder_path:
            human_gender = "male"
        for fn in os.listdir(folder_path):
            parts = parse_filename(fn)
            if parts:
                full_cat = f"human_{human_gender}" if human_gender else category
                trials.append({
                    "path": os.path.join(folder_path, fn),
                    "category": full_cat,
                    "top": parts[0],
                    "bot": parts[1],
                    "congruent": parts[0] == parts[1]
                })
    return trials


# ========================
# IDENTITY ORGANIZATION
# ========================
def organize_by_identity_with_repetitions(trials, repetitions):
    """Organize trials by identity and create multiple copies for repetitions"""
    identity_data = defaultdict(list)
    for trial in trials:
        # Add multiple copies of each stimulus path
        for _ in range(repetitions):
            identity_data[trial['top']].append(trial['path'])

    # Shuffle each identity's stimulus list to randomize repetition order
    for identity in identity_data:
        random.shuffle(identity_data[identity])

    return identity_data


# ========================
# SEQUENCE GENERATION
# ========================
def generate_boolean_sequence(num_trials):
    """Generate same/different sequence for one block"""
    half = num_trials // 2
    booleans = [True] * half + [False] * half

    # Add extra trial if odd number
    if num_trials % 2 == 1:
        booleans.append(random.choice([True, False]))

    random.shuffle(booleans)
    return booleans


def generate_stimulus_sequence(booleans, identity_data, log_file=None):
    """
    Generate stimulus sequence for one category with repetitions
    Returns: list of image paths
    """
    sequence = []
    available_images = copy.deepcopy(identity_data)
    identities = list(available_images.keys())

    if not identities:
        return []

    # Select starting identity
    current_identity = random.choice(identities)
    first_stimulus = available_images[current_identity].pop(0)
    sequence.append(first_stimulus)

    # Debug info
    debug_info = [f"Start: {current_identity} - {os.path.basename(first_stimulus)}"]

    # Track statistics
    same_attempts = 0
    diff_attempts = 0
    same_successes = 0
    diff_successes = 0
    fallbacks = 0

    for is_same in booleans:
        if is_same:
            same_attempts += 1
            # Try to stay with same identity
            if available_images[current_identity]:
                stimulus = available_images[current_identity].pop(0)
                same_successes += 1
                debug_info.append(f"Same: {current_identity} - {os.path.basename(stimulus)}")
            else:
                # Switch if no images left for current identity
                other_ids = [id for id in identities if id != current_identity and available_images[id]]
                if other_ids:
                    current_identity = random.choice(other_ids)
                    stimulus = available_images[current_identity].pop(0)
                    fallbacks += 1
                    debug_info.append(f"Switch (fallback): {current_identity} - {os.path.basename(stimulus)}")
                else:
                    debug_info.append("No images available - skipping")
                    continue
        else:
            diff_attempts += 1
            # Try to switch to different identity
            other_ids = [id for id in identities if id != current_identity and available_images[id]]
            if other_ids:
                current_identity = random.choice(other_ids)
                stimulus = available_images[current_identity].pop(0)
                diff_successes += 1
                debug_info.append(f"Switch: {current_identity} - {os.path.basename(stimulus)}")
            else:
                # Stay if no other identities available
                if available_images[current_identity]:
                    stimulus = available_images[current_identity].pop(0)
                    fallbacks += 1
                    debug_info.append(f"Same (fallback): {current_identity} - {os.path.basename(stimulus)}")
                else:
                    debug_info.append("No images available - skipping")
                    continue

        sequence.append(stimulus)

    # Log sequence info with statistics
    if log_file:
        log_message(log_file, "\n" + "=" * 50)
        log_message(log_file, "SEQUENCE GENERATION STATISTICS:")
        if same_attempts > 0:
            log_message(log_file,
                        f"Same attempts: {same_attempts}, successes: {same_successes} ({same_successes / same_attempts * 100:.1f}%)")
        else:
            log_message(log_file, "Same attempts: 0")

        if diff_attempts > 0:
            log_message(log_file,
                        f"Different attempts: {diff_attempts}, successes: {diff_successes} ({diff_successes / diff_attempts * 100:.1f}%)")
        else:
            log_message(log_file, "Different attempts: 0")

        log_message(log_file, f"Fallbacks: {fallbacks}")
        log_message(log_file, f"Final sequence length: {len(sequence)}")
        log_message(log_file, "")
        log_message(log_file, "SEQUENCE DETAILS:")
        for info in debug_info:
            log_message(log_file, info)
        log_message(log_file, "=" * 50 + "\n")

    return sequence

# ========================
# TRIAL GENERATION
# ========================

def evaluate_sequence_balance(sequence, identity_data, log_file=None):
    """
    Evaluate how balanced the stimulus usage is in a sequence.
    Returns a balance score (lower is better) and usage statistics.
    """
    # Count how many times each stimulus is used
    usage_counts = Counter(sequence)

    # Get all available stimuli
    all_stimuli = []
    for identity_stimuli in identity_data.values():
        all_stimuli.extend(identity_stimuli)

    # Calculate statistics
    usage_values = list(usage_counts.values())
    if not usage_values:
        return float('inf'), {}

    min_usage = min(usage_values)
    max_usage = max(usage_values)
    usage_range = max_usage - min_usage

    # Count unused stimuli
    unused_count = len(all_stimuli) - len(usage_counts)

    # Check for consecutive repetitions (3+ times in a row)
    consecutive_penalty = 0
    if len(sequence) >= 3:
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i + 1] == sequence[i + 2]:
                consecutive_penalty += 20  # Heavy penalty for 3+ consecutive

    # Balance score: penalize high range, unused stimuli, and consecutive repetitions
    balance_score = usage_range * 10 + unused_count * 5 + consecutive_penalty

    stats = {
        'min_usage': min_usage,
        'max_usage': max_usage,
        'usage_range': usage_range,
        'unused_count': unused_count,
        'consecutive_penalty': consecutive_penalty,
        'total_stimuli': len(all_stimuli),
        'used_stimuli': len(usage_counts),
        'balance_score': balance_score
    }

    if log_file:
        log_message(log_file, f"  Balance Score: {balance_score}")
        log_message(log_file, f"  Usage Range: {min_usage}-{max_usage} (range: {usage_range})")
        log_message(log_file, f"  Unused Stimuli: {unused_count}/{len(all_stimuli)}")
        if consecutive_penalty > 0:
            log_message(log_file, f"  Consecutive Penalty: {consecutive_penalty}")

    return balance_score, stats


def generate_balanced_stimulus_sequence(booleans, identity_data, max_attempts=MAX_SEQUENCE_ATTEMPTS, log_file=None):
    """
    Generate a balanced stimulus sequence by trying multiple seeds.
    Returns the best sequence found within max_attempts.
    """
    best_sequence = None
    best_score = float('inf')
    best_stats = {}

    original_random_state = random.getstate()

    if log_file:
        log_message(log_file, f"\nTrying {max_attempts} different seeds for balanced sequence...")

    for attempt in range(max_attempts):
        # Set seed for this attempt
        random.seed(attempt)

        # Generate sequence with current seed
        sequence = generate_stimulus_sequence_single_attempt(booleans, identity_data, MAX_CONSECUTIVE_REPETITIONS)

        if not sequence:
            continue

        # Evaluate balance
        score, stats = evaluate_sequence_balance(sequence, identity_data)

        # Keep track of best sequence
        if score < best_score:
            best_score = score
            best_sequence = sequence.copy()
            best_stats = stats.copy()

            if log_file:
                log_message(log_file, f"  Attempt {attempt}: New best score {score}")

        # If we find a perfect balance, stop early
        if score == 0:
            if log_file:
                log_message(log_file, f"  Perfect balance found at attempt {attempt}")
            break

    # Restore original random state
    random.setstate(original_random_state)

    if log_file:
        log_message(log_file, f"\nBest sequence found:")
        log_message(log_file, f"  Final Balance Score: {best_score}")
        log_message(log_file, f"  Usage Range: {best_stats['min_usage']}-{best_stats['max_usage']}")
        log_message(log_file, f"  Unused Stimuli: {best_stats['unused_count']}/{best_stats['total_stimuli']}")

    return best_sequence if best_sequence else []


def generate_stimulus_sequence_single_attempt(booleans, identity_data, max_consecutive=2):
    """
    Generate stimulus sequence for one attempt (used internally by balanced generator).
    This is a modified version with better balancing and consecutive repetition prevention.
    """
    if not identity_data:
        return []

    sequence = []
    available_images = copy.deepcopy(identity_data)
    identities = list(available_images.keys())

    # Create balanced stimulus pool (trying to ensure balanced distrubtion basically)
    all_stimuli = []
    stimulus_to_identity = {}
    for identity, stimuli in available_images.items():
        for stimulus in stimuli:
            all_stimuli.append(stimulus)
            stimulus_to_identity[stimulus] = identity

    # Create more balanced distribution by cycling through identities
    stimulus_pool = []
    max_per_identity = max(len(stimuli) for stimuli in available_images.values())
    for round_num in range(max_per_identity):
        identity_order = list(available_images.keys())
        random.shuffle(identity_order)
        for identity in identity_order:
            if round_num < len(available_images[identity]):
                stimulus_pool.append(available_images[identity][round_num])

    # Shuffle the balanced pool
    random.shuffle(stimulus_pool)

    # Select first stimulus randomly
    if not stimulus_pool:
        return []

    first_stimulus = random.choice(stimulus_pool)
    current_identity = stimulus_to_identity[first_stimulus]
    stimulus_pool.remove(first_stimulus)
    sequence.append(first_stimulus)

    for is_same in booleans:
        if not stimulus_pool:
            break

        # Check how many consecutive times the same identity appeared
        consecutive_count = 1
        if len(sequence) >= 2:
            last_identity = stimulus_to_identity[sequence[-1]]  # Get identity of last stimulus
            for i in range(len(sequence) - 2, -1, -1):
                if stimulus_to_identity[sequence[i]] == last_identity:
                    consecutive_count += 1
                else:
                    break

        if is_same:
            # Try to find stimulus from same identity
            same_identity_stimuli = [s for s in stimulus_pool
                                     if stimulus_to_identity[s] == current_identity]

            # If we're at max consecutive, avoid the same identity
            if consecutive_count >= max_consecutive:
                last_identity = stimulus_to_identity[sequence[-1]]
                same_identity_stimuli = [s for s in same_identity_stimuli
                                         if stimulus_to_identity[s] != last_identity]

            if same_identity_stimuli:
                # Choose from same identity stimuli (excluding recent repeats)
                chosen = random.choice(same_identity_stimuli)
            else:
                # Fallback: chooses any available stimulus (but not the same one if possible)
                available_choices = [s for s in stimulus_pool if s != sequence[-1]]
                if available_choices:
                    chosen = random.choice(available_choices)
                else:
                    # Last resort: any stimulus
                    chosen = random.choice(stimulus_pool)
                current_identity = stimulus_to_identity[chosen]
        else:
            # Try to find stimulus from different identity
            different_identity_stimuli = [s for s in stimulus_pool
                                          if stimulus_to_identity[s] != current_identity]

            # If wr are at max consecutive, avoid the same identity
            if consecutive_count >= max_consecutive:
                last_identity = stimulus_to_identity[sequence[-1]]
                different_identity_stimuli = [s for s in different_identity_stimuli
                                              if stimulus_to_identity[s] != last_identity]

            if different_identity_stimuli:
                # Choose from different identity stimuli (excluding recent repeats)
                chosen = random.choice(different_identity_stimuli)
                current_identity = stimulus_to_identity[chosen]
            else:
                # Fallback: choose any available stimulus (but not the same one if possible)
                available_choices = [s for s in stimulus_pool if s != sequence[-1]]
                if available_choices:
                    chosen = random.choice(available_choices)
                    current_identity = stimulus_to_identity[chosen]
                else:
                    # Last resort: any stimulus can be chosen
                    chosen = random.choice(stimulus_pool)
                    current_identity = stimulus_to_identity[chosen]

        stimulus_pool.remove(chosen)
        sequence.append(chosen)

    return sequence


def generate_balanced_block_trials(human_trials, monkey_trials, num_trials, condition_name, log_file=None):
    """
    Generate interleaved trials for one block with balanced stimulus usage.
    """
    # Update monkey trial categories to include condition
    updated_monkey_trials = []
    for trial in monkey_trials:
        updated_trial = trial.copy()
        updated_trial['category'] = f"monkey_{condition_name}"  # e.g., monkey_aligned
        updated_monkey_trials.append(updated_trial)

    # Organise by identity with repetitions
    human_data = organize_by_identity_with_repetitions(human_trials, STIMULUS_REPETITIONS)
    monkey_data = organize_by_identity_with_repetitions(updated_monkey_trials, STIMULUS_REPETITIONS)

    # Log available stimuli
    if log_file:
        log_message(log_file, f"Human identities: {list(human_data.keys())}")
        log_message(log_file, f"Human stimuli per identity: {[len(stimuli) for stimuli in human_data.values()]}")
        log_message(log_file, f"Monkey identities: {list(monkey_data.keys())}")
        log_message(log_file, f"Monkey stimuli per identity: {[len(stimuli) for stimuli in monkey_data.values()]}")

    # Generate boolean sequences
    human_booleans = generate_boolean_sequence(num_trials)
    monkey_booleans = generate_boolean_sequence(num_trials)

    if log_file:
        log_message(log_file, f"Human boolean sequence: {human_booleans}")
        log_message(log_file, f"Monkey boolean sequence: {monkey_booleans}")

    # Generate balanced stimulus sequences
    if log_file:
        log_message(log_file, "\nGenerating balanced HUMAN sequence:")
    human_sequence = generate_balanced_stimulus_sequence(human_booleans, human_data, MAX_SEQUENCE_ATTEMPTS, log_file=log_file)

    if log_file:
        log_message(log_file, "\nGenerating balanced MONKEY sequence:")
    monkey_sequence = generate_balanced_stimulus_sequence(monkey_booleans, monkey_data, MAX_SEQUENCE_ATTEMPTS, log_file=log_file)

    # Create path to trial mapping with updated monkey trials (not original ones)
    path_to_trial = {}
    for trial in human_trials + updated_monkey_trials:  # Use updated_monkey_trials here
        path_to_trial[trial['path']] = trial

    # Interleave human and monkey trials
    interleaved = []
    min_length = min(len(human_sequence), len(monkey_sequence))

    for i in range(min_length):
        interleaved.append(path_to_trial[human_sequence[i]])
        interleaved.append(path_to_trial[monkey_sequence[i]])

    # Add any remaining trials
    if len(human_sequence) > min_length:
        for path in human_sequence[min_length:]:
            interleaved.append(path_to_trial[path])

    if len(monkey_sequence) > min_length:
        for path in monkey_sequence[min_length:]:
            interleaved.append(path_to_trial[path])

    # Log trial list
    if log_file:
        log_message(log_file, "\n" + "=" * 50)
        log_message(log_file, "TRIAL ORDER FOR BLOCK:")
        for i, trial in enumerate(interleaved):
            log_message(log_file, f"{i + 1}: {os.path.basename(trial['path'])} - {trial['category']}")
        log_message(log_file, "=" * 50 + "\n")

    return interleaved

# ========================
# DISPLAY UTILITIES
# ========================
def show_instructions(win):
    txt = ("Welcome!\nIs the TOP HALF of the face SAME as the previous same category face?"
           "\n(Monkey with Monkey, Human with Human)\n(Left Arrow = NO / Right Arrow = YES)\nPress ENTER to Start")
    stim = visual.TextStim(win, text=txt,
                           height=INSTRUCTION_TEXT_SIZE,
                           color=INSTRUCTION_TEXT_COLOR,
                           wrapWidth=1000)
    stim.draw()
    win.flip()
    event.waitKeys(keyList=["return", "enter"])


def show_break_screen(win, current_block, total_blocks):
    txt = f"Block {current_block}/{total_blocks} done\nTake a short break\nPress ENTER to Continue"
    stim = visual.TextStim(win, text=txt,
                           height=BREAK_TEXT_SIZE,
                           color=BREAK_TEXT_COLOR)
    stim.draw()
    win.flip()
    event.waitKeys(keyList=["return", "enter"])


# ========================
# EXPERIMENT CORE
# ========================
def run_experiment():
    # Get subject ID only
    info = {"Subject ID": ""}
    if not gui.DlgFromDict(info).OK:
        core.quit()

    sid = info["Subject ID"]

    # Setup logging
    log_file = setup_logging(sid)
    log_message(log_file, f"=== EXPERIMENT LOG FOR SUBJECT {sid} ===")
    log_message(log_file, f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(log_file, f"Stimulus repetitions: {STIMULUS_REPETITIONS}")
    log_message(log_file, f"Trials per sub-block: {NUM_TRIALS_PER_SUB_BLOCK}")
    log_message(log_file, f"Sub-blocks per main block: {SUB_BLOCKS_PER_MAIN_BLOCK}")
    log_message(log_file, "")

    # Load all monkey faces
    monkey_aligned = load_category(os.path.join(BASE_PATH, "monkey_faces/aligned"), "monkey")
    monkey_misaligned = load_category(os.path.join(BASE_PATH, "monkey_faces/misaligned"), "monkey")

    # Split monkey faces by gender block
    random.shuffle(monkey_aligned)
    random.shuffle(monkey_misaligned)
    monkey_pools = {
        'female': {
            'aligned': monkey_aligned[:len(monkey_aligned) // 2],
            'misaligned': monkey_misaligned[:len(monkey_misaligned) // 2]
        },
        'male': {
            'aligned': monkey_aligned[len(monkey_aligned) // 2:],
            'misaligned': monkey_misaligned[len(monkey_misaligned) // 2:]
        }
    }

    noise_folder = os.path.join(BASE_PATH, "noises")
    noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder)
                   if os.path.isfile(os.path.join(noise_folder, f))]

    # intialise results
    results = []

    # Create window with monitor specification
    mon = monitors.Monitor('testMonitor', width=53.0, distance=70.0)
    mon.setSizePix(SCREEN_SIZE)
    win = visual.Window(
        size=SCREEN_SIZE,
        fullscr=True,
        color='white',
        units='pix',
        monitor=mon
    )

    show_instructions(win)

    # Randomize block order for each participant
    randomized_conditions = BLOCK_CONDITIONS.copy()
    random.shuffle(randomized_conditions)

    log_message(log_file, f"Block order: {randomized_conditions}")
    log_message(log_file, "")

    try:
        # Run each main block
        for main_block_idx, (h_folder, m_folder) in enumerate(randomized_conditions, start=1):
            gender = 'female' if 'female' in h_folder else 'male'
            m_condition = m_folder.split('/')[-1]  # 'aligned' or 'misaligned'
            condition_name = f"{gender}_{m_condition}"

            # Load human and monkey faces for this condition
            human = load_category(os.path.join(BASE_PATH, h_folder), f"human_{gender}")
            monkey = monkey_pools[gender][m_condition]

            # Log main block info
            log_message(log_file, f"\n{'=' * 50}")
            log_message(log_file,
                        f"MAIN BLOCK {main_block_idx}: {gender.capitalize()} Human + {m_condition.capitalize()} Monkey")
            log_message(log_file, f"Human stimuli: {len(human)}")
            log_message(log_file, f"Monkey stimuli: {len(monkey)}")
            log_message(log_file, f"Repetitions per stimulus: {STIMULUS_REPETITIONS}")
            log_message(log_file, f"{'=' * 50}\n")

            # Run 3 sub-blocks within this main block
            for sub_block_idx in range(1, SUB_BLOCKS_PER_MAIN_BLOCK + 1):
                log_message(log_file, f"\nStarting SUB-BLOCK {sub_block_idx} of MAIN BLOCK {main_block_idx}")

                # Generate trials for this sub-block (10 per category)
                trials = generate_balanced_block_trials(
                    human,
                    monkey,
                    num_trials=NUM_TRIALS_PER_SUB_BLOCK,
                    condition_name=m_condition,  # Pass the condition name
                    log_file=log_file
                )

                # Track previous top for each category 
                prev = {
                    f"human_{gender}": None,
                    f"monkey_{m_condition}": None  # Use specific monkey condition
                }

                # Track trial counts per category 
                category_counts = {
                    f"human_{gender}": 0,
                    f"monkey_{m_condition}": 0  # Use specific monkey condition
                }

                stim_h = SCREEN_SIZE[1] * STIM_HEIGHT_PCT

                for trial_idx, t in enumerate(trials, 1):
                    # 1) show noise
                    fixation_time = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                    visual.ImageStim(win, image=random.choice(noise_files), size=(None, stim_h)).draw()
                    win.flip()
                    core.wait(fixation_time)

                    # 2) show stimulus for fixed time
                    img_stim = visual.ImageStim(win, image=t['path'], size=(None, stim_h))
                    img_stim.draw()
                    win.flip()
                    core.wait(IMAGE_DISPLAY_TIME)

                    event.clearEvents(eventType='keyboard')

                    # Update category counts
                    cat = t['category']
                    category_counts[cat] += 1
                    current_top = t['top']
                    current_bot = t['bot']

                    # 3) clear screen and prompt for response (only after 2nd trial in category)
                    win.flip()  # blank

                    if category_counts[cat] > 1:
                        prompt = visual.TextStim(win,
                                                 text="TOP HALF SAME? \nLeft Arrow = NO / Right Arrow = YES",
                                                 height=INSTRUCTION_TEXT_SIZE,
                                                 color=INSTRUCTION_TEXT_COLOR,
                                                 wrapWidth=1000)
                        prompt.draw()
                        win.flip()

                        # wait for response
                        resp_clock = core.Clock()
                        response, rt = 'NA', 'NA'
                        responded = False
                        while resp_clock.getTime() < RESPONSE_TIME:
                            keys = event.getKeys(keyList=["right", "left", "escape"], timeStamped=resp_clock)
                            if keys:
                                key, rt = keys[0]
                                responded = True
                                if key == 'escape':
                                    save_data(sid, results, autosave=True)
                                    win.close()
                                    core.quit()
                                response = key
                                break
                    else:
                        # For first trial in category, wait with no prompt
                        core.wait(WAIT_FIRST_TWO)
                        response, rt, responded = 'NA', 'NA', False

                    # Determine if top is same as previous
                    prev_top = prev.get(cat)
                    if prev_top is None:
                        top_same = "NA"  # First trial in category
                    else:
                        top_same = "SAME" if current_top == prev_top else "DIFF"

                    # Calculate response correctness if applicable
                    correct = 'NA'
                    if category_counts[cat] > 1 and responded:
                        expected_response = "right" if top_same == "SAME" else "left"
                        correct = "CORRECT" if response.lower() == expected_response else "INCORRECT"

                    # Trial recording part
                    results.append({
                        'universal_trial_num': len(results)+1,
                        'subject': sid,
                        'main_block': main_block_idx,
                        'block_condition': f"{gender}_{m_condition}",
                        'sub_block': sub_block_idx,
                        'trial_num': trial_idx,
                        'category': cat,
                        'stimulus': os.path.basename(t['path']),
                        'top': current_top,
                        'bot': current_bot,
                        'top_same': top_same,
                        'response': response,
                        'rt': rt,
                        'correct': correct,
                        'fixation_time': fixation_time
                    })

                    # Update previous top for this category
                    prev[cat] = current_top

                    # Log trial details
                    log_message(log_file, f"Trial {trial_idx}: {os.path.basename(t['path'])}")
                    log_message(log_file, f"  Category: {cat}, Count: {category_counts[cat]}")
                    log_message(log_file, f"  Top: {current_top}, Bot: {current_bot}")
                    log_message(log_file, f"  Prev top: {prev_top}")
                    log_message(log_file, f"  Top same: {top_same}")
                    log_message(log_file, f"  Response: {response}, Correct: {correct}")
                    log_message(log_file, "")

                # Log sub-block summary
                log_sub_block_summary(log_file, main_block_idx, sub_block_idx, results, condition_name)

                # Break between sub-blocks
                if sub_block_idx < SUB_BLOCKS_PER_MAIN_BLOCK:
                    # Break between sub-blocks
                    break_text = (
                        f"Part {sub_block_idx} of 3 completed\n\n"
                        f"Take a short break\n\n"
                        f"Press ENTER to continue to next Part"
                    )
                    break_stim = visual.TextStim(win, text=break_text,
                                                 height=BREAK_TEXT_SIZE,
                                                 color=BREAK_TEXT_COLOR)
                    break_stim.draw()
                    win.flip()
                    event.waitKeys(keyList=["return", "enter"])

            # Log main block summary
            log_main_block_summary(log_file, main_block_idx, results, block_condition=condition_name)

            # Show break after each main block except last
            if main_block_idx < len(randomized_conditions):
                show_break_screen(win, main_block_idx, len(randomized_conditions))

        # Log overall experiment summary
        log_experiment_summary(log_file, results, sid)

        log_message(log_file, f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        save_data(sid, results)
    except Exception as e:
        error_msg = f"Error occurred: {e}\n{traceback.format_exc()}"
        log_message(log_file, error_msg)
        print(error_msg)
        save_data(sid, results, autosave=True)
        # Re-raise exception after logging
        raise
    finally:
        win.close()


# =======================
# SAVE RESULTS
# =======================
def save_data(subject_id, data, autosave=False):
    if not data:
        print("No data to save")
        return

    df = pd.DataFrame(data)

    # Create clean filename
    filename = f"{subject_id}_results"
    if autosave:
        filename += "_autosave"

    # Find next available filename
    counter = 1
    final_filename = f"{filename}.csv"
    while os.path.exists(final_filename):
        final_filename = f"{filename}_{counter}.csv"
        counter += 1

    df.to_csv(final_filename, index=False)
    print(f"Data saved to {final_filename}")


if __name__ == '__main__':
    run_experiment()
