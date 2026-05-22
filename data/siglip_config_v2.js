// Hardcoded SigLIP evaluation thresholds.
// Tag-level weight multipliers are no longer configured here —
// they are encoded directly in each tag name using the :x<N> suffix
// (e.g. "underage_adult_with_child:x20", "underage_boy_mature_woman:x5").
export async function run({ inputs }) {
    const { NextNode } = DEFINITIONS;

    return NextNode.from({
        outputs: {
            config: {
                underage_confidence:      0.6,
                underage_min_score:       0.004,
                auto_trigger_minor_risk:  0.85,
                bestiality_threshold:     0.05,
                human_waste_threshold:    0.05,
                blood_threshold:          0.01,
                death_threshold:          0.01,
                weapons_threshold:        0.0105,
                drugs_threshold:          0.01,
                rape_threshold:           0.01,
                label_threshold:          0.15,
                min_label_display_score:  0.001,
                celebrity_threshold:      0.1,
                copyright_threshold:      0.1,
            }
        }
    });
}
