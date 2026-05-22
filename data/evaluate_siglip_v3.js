function round(value) {
    return Math.round(value * 100000) / 100000;
}

// Probability combination: 1 - \u220f(1 - p)
function combineScores(labels, group) {
    return 1 - group.reduce((prod, label) => prod * (1 - (labels[label] || 0)), 1);
}

function getLabels(groupVals, labelKeys, consume, minScore) {
    return labelKeys
        .map(label => ({
            key: label.replace(new RegExp(`^${consume}`), ''),
            value: groupVals[label]
        }))
        // Always include tags with :x<N> multiplier suffix (needed for simulator re-simulation),
        // even if their score is below the display threshold.
        .filter(item => item.value > 0 && (item.value >= minScore || /:x\d/.test(item.key)))
        .sort((a, b) => b.value - a.value)
        .reduce((obj, item) => {
            obj[item.key] = round(item.value);
            return obj;
        }, {});
}

// Reads the multiplier directly from the tag name suffix ":x<N>".
// Example: "underage_adult_with_child:x20" \u2192 multiplier 20
//          "underage_boy_mature_woman:x5"  \u2192 multiplier 5
//          "underage_girl"                 \u2192 multiplier 1 (no suffix)
function getAdjustedScores(rawAffected) {
    const adjusted = { ...rawAffected };
    for (const [key, value] of Object.entries(adjusted)) {
        const match = key.match(/:x(\d+(?:\.\d+)?)$/);
        if (match) {
            const mult = parseFloat(match[1]);
            adjusted[key] = Math.min(value * mult, 0.999);
        }
    }
    return adjusted;
}

export async function run({ inputs }) {
    const { NextNode } = DEFINITIONS;

    // Read thresholds from config input (with safe fallbacks)
    const cfg = inputs.config ?? {};
    const UNDERAGE_CONFIDENCE       = cfg.underage_confidence       ?? 0.6;
    const UNDERAGE_MIN_SCORE        = cfg.underage_min_score        ?? 0.004;
    const AUTO_TRIGGER_MINOR_RISK   = cfg.auto_trigger_minor_risk   ?? 0.85;
    const BESTIALITY_THRESHOLD      = cfg.bestiality_threshold      ?? 0.05;
    const HUMAN_WASTE_THRESHOLD     = cfg.human_waste_threshold     ?? 0.05;
    const BLOOD_THRESHOLD           = cfg.blood_threshold           ?? 0.1;
    const DEATH_THRESHOLD           = cfg.death_threshold           ?? 0.15;
    const WEAPONS_THRESHOLD         = cfg.weapons_threshold         ?? 0.15;
    const DRUGS_THRESHOLD           = cfg.drugs_threshold           ?? 0.15;
    const RAPE_THRESHOLD            = cfg.rape_threshold            ?? 0.15;
    const LABEL_THRESHOLD           = cfg.label_threshold           ?? 0.15;
    const MIN_LABEL_DISPLAY_SCORE   = cfg.min_label_display_score   ?? 0.001;

    const rawAffected = Object.fromEntries(Object.entries(inputs.labels).filter(([key, value]) => value > 0));
    const affected = getAdjustedScores(rawAffected);

    const labels = [];
    const details = {};

    // ===== UNDERAGE (relative threshold + auto-trigger bypass) =====
    const MINOR_LABELS = Object.keys(affected).filter(k => k.startsWith("underage"));
    const minorRisk = combineScores(affected, MINOR_LABELS);

    const ADULT_LABELS = Object.keys(affected).filter(k => k.startsWith("adult"));
    const adultConfidence = combineScores(affected, ADULT_LABELS);

    const totalUnderageScore = minorRisk + adultConfidence;
    const underageConfidence = totalUnderageScore > 0 ? minorRisk / totalUnderageScore : 0;

    const isAbsolutelyUnderage = minorRisk >= AUTO_TRIGGER_MINOR_RISK;

    if ( (underageConfidence >= UNDERAGE_CONFIDENCE && minorRisk >= UNDERAGE_MIN_SCORE) || isAbsolutelyUnderage ) {
        if (!labels.includes('underage')) {
            labels.push('underage');
        }
    }

    details.underage = {
        minor: round(minorRisk),
        adult: round(adultConfidence),
        confidence: round(underageConfidence),
        thresholds: {
            confidence: UNDERAGE_CONFIDENCE,
            minScore: UNDERAGE_MIN_SCORE,
            autoTriggerLimit: AUTO_TRIGGER_MINOR_RISK
        },
        labels: {
            underage: getLabels(affected, MINOR_LABELS, "underage_", MIN_LABEL_DISPLAY_SCORE),
            adult: getLabels(affected, ADULT_LABELS, "adult_", MIN_LABEL_DISPLAY_SCORE)
        }
    };

    // ===== BESTIALITY =====
    const BESTIALITY_LABELS = Object.keys(affected).filter(k => k.startsWith("bestiality"));
    const bestialityRisk = combineScores(affected, BESTIALITY_LABELS);

    const FURRY_LABELS = Object.keys(affected).filter(k => k.startsWith("furry"));
    const furryConfidence = combineScores(affected, FURRY_LABELS);

    if (bestialityRisk >= BESTIALITY_THRESHOLD && bestialityRisk > furryConfidence) {
        labels.push('bestiality');
    }

    details.bestiality = {
        risk: bestialityRisk.toFixed(5),
        furry: furryConfidence.toFixed(5),
        threshold: BESTIALITY_THRESHOLD,
        labels: {
            bestiality: getLabels(affected, BESTIALITY_LABELS, "bestiality_", MIN_LABEL_DISPLAY_SCORE),
            furry: getLabels(affected, FURRY_LABELS, "furry_", MIN_LABEL_DISPLAY_SCORE)
        }
    };

    // ===== HUMAN WASTE =====
    const HUMAN_WASTE_LABLES = Object.keys(affected).filter(k => k.startsWith("human_waste"));
    const humanWasteRisk = combineScores(affected, HUMAN_WASTE_LABLES);

    const NO_WASTE_LABLES = Object.keys(affected).filter(k => k.startsWith("no_waste"));
    const noWasteConfidence = combineScores(affected, NO_WASTE_LABLES);

    if (humanWasteRisk >= HUMAN_WASTE_THRESHOLD && humanWasteRisk > noWasteConfidence) {
        labels.push('human_waste');
    }

    details.humanWaste = {
        risk: humanWasteRisk.toFixed(5),
        noWaste: noWasteConfidence.toFixed(5),
        threshold: HUMAN_WASTE_THRESHOLD,
        labels: {
            humanWaste: getLabels(affected, HUMAN_WASTE_LABLES, "human_waste_", MIN_LABEL_DISPLAY_SCORE),
            noWaste: getLabels(affected, NO_WASTE_LABLES, "no_waste_", MIN_LABEL_DISPLAY_SCORE),
        }
    };

    // ===== BLOOD =====
    const BLOOD_LABLES = Object.keys(affected).filter(k => k.startsWith("blood"));
    const bloodRisk = combineScores(affected, BLOOD_LABLES);

    const NO_BLOOD_LABLES = Object.keys(affected).filter(k => k.startsWith("no_blood"));
    const noBloodConfidence = combineScores(affected, NO_BLOOD_LABLES);

    if (bloodRisk >= BLOOD_THRESHOLD && bloodRisk > noBloodConfidence) {
        labels.push('blood');
    }

    details.blood = {
        blood: bloodRisk.toFixed(5),
        noBlood: noBloodConfidence.toFixed(5),
        threshold: BLOOD_THRESHOLD,
        labels: {
            blood: getLabels(affected, BLOOD_LABLES, "blood_", MIN_LABEL_DISPLAY_SCORE),
            noBlood: getLabels(affected, NO_BLOOD_LABLES, "no_blood_", MIN_LABEL_DISPLAY_SCORE)
        }
    };

    // ===== DEATH =====
    const DEATH_LABLES = Object.keys(affected).filter(k => k.startsWith("death"));
    const deathRisk = combineScores(affected, DEATH_LABLES);

    const NO_DEATH_LABLES = Object.keys(affected).filter(k => k.startsWith("no_death"));
    const noDeathConfidence = combineScores(affected, NO_DEATH_LABLES);

    if (deathRisk >= DEATH_THRESHOLD && deathRisk > noDeathConfidence) {
        labels.push('death_and_murder');
    }

    details.death = {
        risk: deathRisk.toFixed(5),
        noDeath: noDeathConfidence.toFixed(5),
        threshold: DEATH_THRESHOLD,
        labels: {
            death: getLabels(affected, DEATH_LABLES, "death_", MIN_LABEL_DISPLAY_SCORE),
            noDeath: getLabels(affected, NO_DEATH_LABLES, "no_death_", MIN_LABEL_DISPLAY_SCORE),
        }
    };

    // ===== WEAPONS =====
    const WEAPONS_LABLES = Object.keys(affected).filter(k => k.startsWith("weapons"));
    const weaponsRisk = combineScores(affected, WEAPONS_LABLES);

    const NO_WEAPON_LABLES = Object.keys(affected).filter(k => k.startsWith("no_weapon"));
    const noWeaponConfidence = combineScores(affected, NO_WEAPON_LABLES);

    if (weaponsRisk >= WEAPONS_THRESHOLD && weaponsRisk > noWeaponConfidence) {
        labels.push('weapons_and_military');
    }

    details.weapons = {
        risk: weaponsRisk.toFixed(5),
        noWeapon: noWeaponConfidence.toFixed(5),
        threshold: WEAPONS_THRESHOLD,
        labels: {
            weapons: getLabels(affected, WEAPONS_LABLES, "weapons_", MIN_LABEL_DISPLAY_SCORE),
            noWeapon: getLabels(affected, NO_WEAPON_LABLES, "no_weapon_", MIN_LABEL_DISPLAY_SCORE),
        }
    };

    // ===== DRUGS =====
    const DRUGS_LABLES = Object.keys(affected).filter(k => k.startsWith("drugs"));
    const drugsRisk = combineScores(affected, DRUGS_LABLES);

    const NO_DRUGS_LABLES = Object.keys(affected).filter(k => k.startsWith("no_drugs"));
    const noDrugsConfidence = combineScores(affected, NO_DRUGS_LABLES);

    if (drugsRisk >= DRUGS_THRESHOLD && drugsRisk > noDrugsConfidence) {
        labels.push('drugs');
    }

    details.drugs = {
        risk: drugsRisk.toFixed(5),
        noDrugs: noDrugsConfidence.toFixed(5),
        threshold: DRUGS_THRESHOLD,
        labels: {
            drugs: getLabels(affected, DRUGS_LABLES, "drugs_", MIN_LABEL_DISPLAY_SCORE),
            noDrugs: getLabels(affected, NO_DRUGS_LABLES, "no_drugs_", MIN_LABEL_DISPLAY_SCORE),
        }
    };

    // ===== RAPE =====
    const RAPE_LABLES = Object.keys(affected).filter(k => k.startsWith("rape"));
    const rapeRisk = combineScores(affected, RAPE_LABLES);

    const NO_RAPE_LABLES = Object.keys(affected).filter(k => k.startsWith("no_rape"));
    const noRapeConfidence = combineScores(affected, NO_RAPE_LABLES);

    if (rapeRisk >= RAPE_THRESHOLD && rapeRisk > noRapeConfidence) {
        labels.push('rape');
    }

    details.rape = {
        risk: rapeRisk.toFixed(5),
        noRape: noRapeConfidence.toFixed(5),
        threshold: RAPE_THRESHOLD,
        labels: {
            rape: getLabels(affected, RAPE_LABLES, "rape_", MIN_LABEL_DISPLAY_SCORE),
            noRape: getLabels(affected, NO_RAPE_LABLES, "no_rape_", MIN_LABEL_DISPLAY_SCORE),
        }
    };

    // ===== ETHNICITY =====
    const ASIAN_LABLES = Object.keys(affected).filter(k => k.startsWith("asian"));
    const asianScore = combineScores(affected, ASIAN_LABLES);
    if (asianScore >= LABEL_THRESHOLD) labels.push('asian');
    details.asian = {
        score: asianScore.toFixed(5),
        threshold: LABEL_THRESHOLD,
        labels: getLabels(affected, ASIAN_LABLES, "asian_", MIN_LABEL_DISPLAY_SCORE),
    };

    const EBONY_LABLES = Object.keys(affected).filter(k => k.startsWith("ebony"));
    const ebonyScore = combineScores(affected, EBONY_LABLES);
    if (ebonyScore >= LABEL_THRESHOLD) labels.push('ebony');
    details.ebony = {
        score: ebonyScore.toFixed(5),
        threshold: LABEL_THRESHOLD,
        labels: getLabels(affected, EBONY_LABLES, "ebony_", MIN_LABEL_DISPLAY_SCORE),
    };

    const ARAB_LABLES = Object.keys(affected).filter(k => k.startsWith("arab"));
    const arabScore = combineScores(affected, ARAB_LABLES);
    if (arabScore >= LABEL_THRESHOLD) labels.push('arab');
    details.arab = {
        score: arabScore.toFixed(5),
        threshold: LABEL_THRESHOLD,
        labels: getLabels(affected, ARAB_LABLES, "arab_", MIN_LABEL_DISPLAY_SCORE),
    };

    const LATIN_LABLES = Object.keys(affected).filter(k => k.startsWith("latin"));
    const latinScore = combineScores(affected, LATIN_LABLES);
    if (latinScore >= LABEL_THRESHOLD) labels.push('latin');
    details.latin = {
        score: latinScore.toFixed(5),
        threshold: LABEL_THRESHOLD,
        labels: getLabels(affected, LATIN_LABLES, "latin_", MIN_LABEL_DISPLAY_SCORE),
    };

    const INDIAN_LABLES = Object.keys(affected).filter(k => k.startsWith("indian"));
    const indianScore = combineScores(affected, INDIAN_LABLES);
    if (indianScore >= LABEL_THRESHOLD) labels.push('indian');
    details.indian = {
        score: indianScore.toFixed(5),
        threshold: LABEL_THRESHOLD,
        labels: getLabels(affected, INDIAN_LABLES, "indian_", MIN_LABEL_DISPLAY_SCORE),
    };

    // ===== CELEBRITY (real-person deepfakes / face-swap) =====
    const CELEBRITY_THRESHOLD = cfg.celebrity_threshold ?? 0.1;
    const CELEBRITY_LABELS = Object.keys(affected).filter(k => k.startsWith("celebrity"));
    const celebrityRisk = combineScores(affected, CELEBRITY_LABELS);
    const NO_CELEBRITY_LABELS = Object.keys(affected).filter(k => k.startsWith("no_celebrity"));
    const noCelebrityConfidence = combineScores(affected, NO_CELEBRITY_LABELS);

    if (celebrityRisk >= CELEBRITY_THRESHOLD && celebrityRisk > noCelebrityConfidence) {
        labels.push('celebrity');
    }
    details.celebrity = {
        risk: celebrityRisk.toFixed(5),
        noCelebrity: noCelebrityConfidence.toFixed(5),
        threshold: CELEBRITY_THRESHOLD,
        labels: {
            celebrity: getLabels(affected, CELEBRITY_LABELS, "celebrity_", MIN_LABEL_DISPLAY_SCORE),
            no_celebrity: getLabels(affected, NO_CELEBRITY_LABELS, "no_celebrity_", MIN_LABEL_DISPLAY_SCORE),
        },
    };

    // ===== COPYRIGHT (protected fictional characters / IP) =====
    const COPYRIGHT_THRESHOLD = cfg.copyright_threshold ?? 0.1;
    const COPYRIGHT_LABELS = Object.keys(affected).filter(k => k.startsWith("copyright"));
    const copyrightRisk = combineScores(affected, COPYRIGHT_LABELS);
    const NO_COPYRIGHT_LABELS = Object.keys(affected).filter(k => k.startsWith("no_copyright"));
    const noCopyrightConfidence = combineScores(affected, NO_COPYRIGHT_LABELS);

    if (copyrightRisk >= COPYRIGHT_THRESHOLD && copyrightRisk > noCopyrightConfidence) {
        labels.push('copyright');
    }
    details.copyright = {
        risk: copyrightRisk.toFixed(5),
        noCopyright: noCopyrightConfidence.toFixed(5),
        threshold: COPYRIGHT_THRESHOLD,
        labels: {
            copyright: getLabels(affected, COPYRIGHT_LABELS, "copyright_", MIN_LABEL_DISPLAY_SCORE),
            no_copyright: getLabels(affected, NO_COPYRIGHT_LABELS, "no_copyright_", MIN_LABEL_DISPLAY_SCORE),
        },
    };

    return NextNode.from({
        outputs: {
            labels,
            details
        }
    });
}