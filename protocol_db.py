
from schemas import Protocol

# --- Protocol 1: Relevant to GSE68086 ---
RNA_EXTRACTION_PROTOCOL = {
    "title": "Total RNA Isolation from Platelets",
    "objective": "To extract high-quality total RNA from purified human platelets for downstream applications like RNA sequencing.",
    "materials": {
        "Reagent": "TRIzol Reagent (or similar), Chloroform, Isopropanol, 75% Ethanol (nuclease-free), RNase-free water, Glycogen (RNA-grade carrier)",
        "Equipment": "Microcentrifuge (4°C), Pipettes, Nuclease-free tubes",
        "Sample": "Purified platelet pellet"
    },
    "steps": [
        {"step_number": 1, "description": "Lyse the platelet pellet by adding 1 mL of TRIzol Reagent. Pipette up and down to homogenize.", "duration_minutes": 5},
        {"step_number": 2, "description": "Incubate the homogenate for 5 minutes at room temperature to permit complete dissociation of nucleoprotein complexes.", "duration_minutes": 5},
        {"step_number": 3, "description": "Add 0.2 mL of chloroform per 1 mL of TRIzol. Cap the tube securely and shake vigorously for 15 seconds.", "duration_minutes": 2},
        {"step_number": 4, "description": "Incubate for 3 minutes at room temperature, then centrifuge at 12,000 x g for 15 minutes at 4°C.", "duration_minutes": 20},
        {"step_number": 5, "description": "Transfer the upper, aqueous phase (which contains the RNA) to a fresh tube. Be careful not to disturb the interphase.", "duration_minutes": 3},
        {"step_number": 6, "description": "Precipitate the RNA by adding 0.5 mL of isopropanol and 1 µL of glycogen. Mix and incubate for 10 minutes at room temperature.", "duration_minutes": 10},
        {"step_number": 7, "description": "Centrifuge at 12,000 x g for 10 minutes at 4°C. A small white pellet of RNA should be visible.", "duration_minutes": 12},
        {"step_number": 8, "description": "Discard the supernatant. Wash the RNA pellet with 1 mL of 75% ethanol.", "duration_minutes": 5},
        {"step_number": 9, "description": "Centrifuge at 7,500 x g for 5 minutes at 4°C. Discard the ethanol wash.", "duration_minutes": 7},
        {"step_number": 10, "description": "Briefly air-dry the pellet for 5-10 minutes. Do not over-dry. Resuspend the RNA in 20 µL of RNase-free water.", "duration_minutes": 10}
    ]
}

# --- Protocol 2: Relevant to GSE68086 ---
RNA_SEQ_LIBRARY_PREP_PROTOCOL = {
    "title": "Illumina-Compatible RNA-Seq Library Preparation",
    "objective": "To convert total RNA into a cDNA library suitable for Illumina next-generation sequencing (NGS).",
    "materials": {
        "Kit": "Illumina TruSeq Stranded Total RNA Library Prep Kit (or equivalent)",
        "Reagent": "Extracted Total RNA, Nuclease-free water",
        "Equipment": "Thermocycler, Pipettes, Magnetic stand, Qubit fluorometer, Bioanalyzer"
    },
    "steps": [
        {"step_number": 1, "description": "Deplete ribosomal RNA (rRNA) from 1 µg of total RNA using Ribo-Zero beads as per the manufacturer's instructions.", "duration_minutes": 60},
        {"step_number": 2, "description": "Fragment the rRNA-depleted RNA using heat and a fragmentation buffer to generate appropriately sized pieces.", "duration_minutes": 15},
        {"step_number": 3, "description": "Synthesize the first strand of cDNA using reverse transcriptase and random primers.", "duration_minutes": 30},
        {"step_number": 4, "description": "Synthesize the second strand of cDNA, incorporating dUTP to ensure strand specificity.", "duration_minutes": 60},
        {"step_number": 5, "description": "Perform end-repair on the double-stranded cDNA to create blunt ends.", "duration_minutes": 30},
        {"step_number": 6, "description": "Adenylate the 3' ends of the cDNA fragments to prepare them for adapter ligation.", "duration_minutes": 30},
        {"step_number": 7, "description": "Ligate the indexed Illumina sequencing adapters to the cDNA fragments.", "duration_minutes": 30},
        {"step_number": 8, "description": "Clean up the ligation products using magnetic beads to remove excess adapters.", "duration_minutes": 15},
        {"step_number": 9, "description": "Amplify the cDNA library via PCR (10-15 cycles) to enrich for adapter-ligated fragments.", "duration_minutes": 30},
        {"step_number": 10, "description": "Validate the final library. Check its size distribution on a Bioanalyzer and quantify its concentration using a Qubit fluorometer before sequencing.", "duration_minutes": 45}
    ]
}

def load_protocol_library():
    """Simulates loading verified protocols from a database."""
    library = {
        "rna isolation": Protocol(**RNA_EXTRACTION_PROTOCOL),
        "rna extraction": Protocol(**RNA_EXTRACTION_PROTOCOL),
        "library preparation": Protocol(**RNA_SEQ_LIBRARY_PREP_PROTOCOL),
        "rna-seq": Protocol(**RNA_SEQ_LIBRARY_PREP_PROTOCOL)
    }
    return library