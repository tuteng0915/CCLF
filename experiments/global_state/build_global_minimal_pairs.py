"""EXP-GS9: hand-constructed minimal contrast pairs.

Each pair shares an IDENTICAL local suffix (the words immediately before the
target slot are byte-for-byte the same string in context_A and context_B);
only the early frame-setting clause differs. target_A/target_B are the two
frame-appropriate completions. See docs/specs/EXP-GS9-spec.md Section 0-1 for
the design rationale and the experimenter-bias caveat.
"""

PAIRS = [
    dict(domain="disaster_election_1",
         context_A="After the earthquake, the region experienced a wave of",
         context_B="After the election, the region experienced a wave of",
         target_A="destruction", target_B="celebration"),
    dict(domain="disaster_election_2",
         context_A="Following the hurricane, local authorities called for immediate",
         context_B="Following the election, local authorities called for immediate",
         target_A="evacuation", target_B="reform"),
    dict(domain="sports_finance_1",
         context_A="After the championship game, everyone in the room began to",
         context_B="After the earnings call, everyone in the room began to",
         target_A="cheer", target_B="worry"),
    dict(domain="sports_finance_2",
         context_A="After the final match, the spokesperson spoke publicly about the recent",
         context_B="After the market crash, the spokesperson spoke publicly about the recent",
         target_A="victory", target_B="losses"),
    dict(domain="weather_health_1",
         context_A="Because of the storm warning, residents were told to stay",
         context_B="Because of the disease outbreak, residents were told to stay",
         target_A="indoors", target_B="isolated"),
    dict(domain="weather_health_2",
         context_A="As the heat wave continued, the city issued a public",
         context_B="As the flu season continued, the city issued a public",
         target_A="advisory", target_B="warning"),
    dict(domain="crime_tech_1",
         context_A="After the robbery was reported, the team spent the day collecting",
         context_B="After the software bug was reported, the team spent the day collecting",
         target_A="evidence", target_B="logs"),
    dict(domain="crime_tech_2",
         context_A="Following the arrest, the spokesperson gave a statement about the new",
         context_B="Following the product launch, the spokesperson gave a statement about the new",
         target_A="charges", target_B="pricing"),
    dict(domain="politics_entertainment_1",
         context_A="Before the vote, the speaker delivered a speech about",
         context_B="Before the premiere, the speaker delivered a speech about",
         target_A="policy", target_B="filmmaking"),
    dict(domain="politics_entertainment_2",
         context_A="In the committee meeting, everyone spent the morning discussing the new",
         context_B="In the movie theater, everyone spent the morning discussing the new",
         target_A="legislation", target_B="movie"),
    dict(domain="science_dialogue_1",
         context_A="In the research paper, they carefully described their",
         context_B="In their conversation, they carefully described their",
         target_A="findings", target_B="feelings"),
    dict(domain="science_dialogue_2",
         context_A="In the laboratory, they carefully recorded the",
         context_B="In the kitchen, they carefully recorded the",
         target_A="measurements", target_B="recipe"),
]
