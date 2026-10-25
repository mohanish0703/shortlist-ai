"""Domain knowledge shared by the generator, the parser and the matcher.

Canonical names live here (skills, certificates, cities, languages) together with the messy
ways people write them. The generator renders the messy forms; the parser maps them back.
All employer, school and person names are invented.
"""

from __future__ import annotations

from datetime import date

from shortlist.schema import JobRequirement

# "Present" / "till date" in a resume is counted up to this date.
REFERENCE_DATE = date(2026, 10, 1)

# Each role: titles that count as the same job, related titles, canonical skills with the ways
# candidates write them, certificates, a typical monthly salary range (INR) and a Hinglish phrase.
ROLES: dict[str, dict] = {
    "housekeeping": {
        "titles": [
            "Housekeeping Staff",
            "Housekeeping Attendant",
            "Room Attendant",
            "Cleaning Staff",
        ],
        "adjacent_titles": ["Office Boy", "Kitchen Helper", "Laundry Assistant"],
        "skills": {
            "room cleaning": ["room cleaning", "cleaning of rooms", "room servicing"],
            "floor mopping": ["mopping", "floor cleaning", "floor mopping"],
            "laundry": ["laundry", "washing and ironing", "linen handling"],
            "bathroom sanitation": ["toilet cleaning", "bathroom cleaning", "washroom sanitation"],
            "chemical handling": ["use of cleaning chemicals", "chemical dilution"],
            "bed making": ["bed making", "bedsheet changing"],
            "machine scrubbing": ["scrubbing machine", "floor scrubber operation"],
        },
        "certifications": [
            "PMKVY Housekeeping Attendant",
            "First Aid Training",
            "Fire Safety Training",
        ],
        "salary": (12_000, 20_000),
        "hinglish": "housekeeping ka kaam",
        "employers": [
            "Hotel Sai Residency",
            "Sparkle Facility Services",
            "Green Valley Apartments",
            "Lotus Multispeciality Hospital",
            "CleanPro Services Pvt Ltd",
        ],
    },
    "front_desk": {
        "titles": [
            "Receptionist",
            "Front Desk Executive",
            "Front Office Assistant",
            "Guest Relations Executive",
        ],
        "adjacent_titles": ["Telecaller", "Office Assistant", "Data Entry Operator"],
        "skills": {
            "guest check-in": ["guest check-in", "check-in / check-out", "guest registration"],
            "phone handling": ["phone handling", "attending calls", "EPABX handling"],
            "MS Office": ["MS Office", "MS Word & Excel", "computer basics (Word, Excel)"],
            "visitor management": ["visitor management", "visitor register", "managing visitors"],
            "billing": ["billing", "cash & card billing", "invoice preparation"],
            "complaint handling": ["complaint handling", "handling guest complaints"],
            "email correspondence": ["email correspondence", "replying to mails"],
        },
        "certifications": ["Diploma in Hotel Management", "MS Office Certificate"],
        "salary": (15_000, 28_000),
        "hinglish": "reception ka kaam",
        "employers": [
            "Hotel Grand Orchid",
            "Sunrise Dental Clinic",
            "Apex Business Centre",
            "Silver Oak Inn",
            "Medilife Diagnostics",
        ],
    },
    "security": {
        "titles": ["Security Guard", "Security Officer", "Security Supervisor"],
        "adjacent_titles": ["Parking Attendant", "Gate Keeper", "Office Boy"],
        "skills": {
            "access control": ["access control", "gate duty", "entry-exit control"],
            "CCTV monitoring": ["CCTV monitoring", "CCTV camera watching", "control room duty"],
            "patrolling": ["patrolling", "night patrolling", "round duty"],
            "visitor management": ["visitor register", "visitor management"],
            "fire safety": ["fire safety", "fire extinguisher use"],
            "emergency response": ["emergency response", "emergency handling"],
        },
        "certifications": ["PSARA Security Training", "Fire Safety Training", "First Aid Training"],
        "salary": (14_000, 22_000),
        "hinglish": "security guard ki duty",
        "employers": [
            "Shield Security Services",
            "Eagle Eye Protection Pvt Ltd",
            "Prestige Tech Park",
            "Harmony Residency",
            "Vigil Guards Agency",
        ],
    },
    "delivery": {
        "titles": ["Delivery Executive", "Delivery Associate", "Delivery Boy", "Bike Rider"],
        "adjacent_titles": ["Driver", "Warehouse Helper", "Courier Helper"],
        "skills": {
            "two-wheeler riding": ["two-wheeler riding", "bike riding", "2 wheeler driving"],
            "route navigation": ["route navigation", "knows city routes", "area knowledge"],
            "cash handling": ["cash handling", "COD cash collection", "cash collection"],
            "Google Maps": ["Google Maps", "GPS / maps app"],
            "customer handling": ["customer handling", "polite with customers"],
            "parcel handling": [
                "parcel handling",
                "package sorting",
                "loading / unloading parcels",
            ],
        },
        "certifications": ["Two-Wheeler Driving Licence", "LMV Driving Licence"],
        "salary": (15_000, 25_000),
        "hinglish": "delivery ka kaam",
        "employers": [
            "QuickDrop Logistics",
            "FreshKart Grocery",
            "SpeedyParcel Couriers",
            "Annapoorna Foods",
            "MetroMove Logistics",
        ],
    },
    "customer_support": {
        "titles": [
            "Customer Support Executive",
            "Customer Care Executive",
            "Call Centre Agent",
            "CSR",
        ],
        "adjacent_titles": ["Telecaller", "Data Entry Operator", "Receptionist"],
        "skills": {
            "inbound calls": ["inbound calls", "inbound voice process", "handling customer calls"],
            "complaint handling": ["complaint handling", "complaint resolution"],
            "CRM software": ["CRM software", "Salesforce / CRM tools", "CRM entry"],
            "typing": ["typing", "fast typing (35 wpm)", "typing speed 40 wpm"],
            "email support": ["email support", "non-voice (email) process"],
            "ticketing": ["ticketing", "ticket raising", "Zendesk-like ticket tools"],
        },
        "certifications": ["MS Office Certificate", "Typing Certificate"],
        "salary": (16_000, 30_000),
        "hinglish": "call centre mein customer support ka kaam",
        "employers": [
            "Vertex BPO Solutions",
            "CallServe India",
            "Nimbus Telecom Support",
            "BrightPath Services",
            "Omni CX Pvt Ltd",
        ],
    },
}

GENERIC_SKILLS: dict[str, list[str]] = {
    "teamwork": ["teamwork", "team player", "work in team"],
    "punctuality": ["punctuality", "punctual", "always on time"],
    "hardworking": ["hardworking", "hard working", "sincere and hardworking"],
}

# canonical skill -> every way it is written, across roles
SKILL_FORMS: dict[str, list[str]] = {}
for _spec in [*(r["skills"] for r in ROLES.values()), GENERIC_SKILLS]:
    for _skill, _forms in _spec.items():
        _merged = SKILL_FORMS.setdefault(_skill, [])
        _merged.extend(f for f in _forms if f not in _merged)

SKILL_VOCAB: list[str] = sorted(SKILL_FORMS)

# education level -> ways it is written
EDUCATION_FORMS: dict[str, list[str]] = {
    "Below 10th": ["8th pass", "9th standard", "Below SSLC"],
    "10th": ["10th pass", "SSLC", "Matriculation", "SSC"],
    "12th": ["12th pass", "PUC", "HSC", "Intermediate"],
    "ITI": ["ITI", "ITI (Electrician)", "ITI (Fitter)"],
    "Diploma": ["Diploma", "Polytechnic Diploma"],
    "Graduate": ["B.Com", "BA", "B.Sc", "BBA"],
    "Postgraduate": ["M.Com", "MA", "MBA"],
}

# canonical certificate -> ways it is written
CERTIFICATIONS: dict[str, list[str]] = {
    "PMKVY Housekeeping Attendant": [
        "PMKVY Housekeeping Attendant certificate",
        "Housekeeping Attendant (PMKVY)",
        "PMKVY certified - housekeeping",
    ],
    "First Aid Training": ["First Aid training", "Basic First Aid certificate", "First-aid course"],
    "Fire Safety Training": [
        "Fire Safety training",
        "Fire & Safety course",
        "Fire fighting basic training",
    ],
    "PSARA Security Training": [
        "PSARA security guard training",
        "Private Security Training (PSARA)",
        "PSARA certified",
    ],
    "Two-Wheeler Driving Licence": [
        "Two wheeler driving licence",
        "DL - 2 wheeler (MCWG)",
        "Bike licence",
    ],
    "LMV Driving Licence": ["LMV licence", "Driving licence (LMV - car)", "4 wheeler DL"],
    "Diploma in Hotel Management": [
        "Diploma in Hotel Management",
        "DHM (1 yr)",
        "Hotel Management diploma",
    ],
    "MS Office Certificate": ["MS Office certificate", "MS-CIT", "Computer course (MS Office)"],
    "Typing Certificate": [
        "Typing certificate (35 wpm)",
        "English typing certificate",
        "Typing course certificate",
    ],
}

# canonical city -> (state language, ways it is written)
CITIES: dict[str, tuple[str, list[str]]] = {
    "Bengaluru": (
        "Kannada",
        [
            "Bengaluru",
            "Bangalore",
            "B'lore",
            "BLR",
            "Whitefield, Bangalore",
            "Electronic City, Bengaluru",
        ],
    ),
    "Pune": ("Marathi", ["Pune", "Poona", "Hinjewadi, Pune", "Kothrud, Pune"]),
    "Hyderabad": ("Telugu", ["Hyderabad", "Hyd", "Kukatpally, Hyderabad", "HYDERABAD"]),
    "Chennai": ("Tamil", ["Chennai", "Madras", "Tambaram, Chennai", "T. Nagar, Chennai"]),
    "Noida": ("Hindi", ["Noida", "NOIDA", "Sector 62, Noida", "Noida (UP)"]),
    "Mysuru": ("Kannada", ["Mysuru", "Mysore"]),
    "Tumakuru": ("Kannada", ["Tumakuru", "Tumkur"]),
    "Satara": ("Marathi", ["Satara"]),
    "Lonavala": ("Marathi", ["Lonavala", "Lonavla"]),
    "Warangal": ("Telugu", ["Warangal"]),
    "Sangareddy": ("Telugu", ["Sangareddy"]),
    "Vellore": ("Tamil", ["Vellore"]),
    "Kanchipuram": ("Tamil", ["Kanchipuram", "Kancheepuram"]),
    "Meerut": ("Hindi", ["Meerut"]),
    "Aligarh": ("Hindi", ["Aligarh"]),
    "Patna": ("Hindi", ["Patna", "Patna, Bihar"]),
    "Lucknow": ("Hindi", ["Lucknow", "Lucknow (UP)"]),
    "Kolkata": ("Bengali", ["Kolkata", "Calcutta"]),
    "Bhubaneswar": ("Odia", ["Bhubaneswar", "BBSR"]),
    "Guwahati": ("Assamese", ["Guwahati"]),
    "Kochi": ("Malayalam", ["Kochi", "Cochin", "Ernakulam"]),
    "Jaipur": ("Hindi", ["Jaipur"]),
    "Indore": ("Hindi", ["Indore"]),
}

# Within commuting / short-relocation range of a job city: a recruiter treats it as borderline.
NEARBY: dict[str, list[str]] = {
    "Bengaluru": ["Mysuru", "Tumakuru"],
    "Pune": ["Satara", "Lonavala"],
    "Hyderabad": ["Warangal", "Sangareddy"],
    "Chennai": ["Vellore", "Kanchipuram"],
    "Noida": ["Meerut", "Aligarh"],
}

FAR_CITIES = ["Patna", "Lucknow", "Kolkata", "Bhubaneswar", "Guwahati", "Kochi", "Jaipur", "Indore"]

LANGUAGE_FORMS: dict[str, list[str]] = {
    "Hindi": ["Hindi"],
    "English": ["English"],
    "Kannada": ["Kannada", "Kanada"],
    "Tamil": ["Tamil"],
    "Telugu": ["Telugu", "Telgu"],
    "Marathi": ["Marathi"],
    "Bengali": ["Bengali", "Bangla"],
    "Odia": ["Odia", "Oriya"],
    "Malayalam": ["Malayalam"],
    "Urdu": ["Urdu"],
    "Assamese": ["Assamese"],
}

# Name pools by home language, so names, home towns and languages stay plausible together.
NAMES: dict[str, tuple[list[str], list[str]]] = {
    "Hindi": (
        [
            "Ramesh",
            "Sunita",
            "Amit",
            "Pooja",
            "Rajesh",
            "Neha",
            "Suresh",
            "Anjali",
            "Vikas",
            "Rekha",
        ],
        ["Kumar", "Devi", "Yadav", "Sharma", "Singh", "Verma", "Gupta", "Paswan", "Mishra"],
    ),
    "Kannada": (
        ["Manjunath", "Shilpa", "Raghavendra", "Kavya", "Nagaraj", "Deepa", "Basavaraj"],
        ["Gowda", "Hegde", "Shetty", "Naik", "Patil", "K", "M"],
    ),
    "Tamil": (
        ["Karthik", "Priya", "Murugan", "Lakshmi", "Senthil", "Divya", "Arun"],
        ["Raja", "Murugan", "Subramanian", "S", "Natarajan", "Pandian"],
    ),
    "Telugu": (
        ["Venkatesh", "Swathi", "Srinivas", "Lavanya", "Ravi", "Padma"],
        ["Rao", "Reddy", "Naidu", "Chowdary", "Varma"],
    ),
    "Marathi": (
        ["Sachin", "Snehal", "Ganesh", "Pallavi", "Sagar", "Ashwini"],
        ["Patil", "Jadhav", "Pawar", "Kulkarni", "Shinde", "More"],
    ),
    "Bengali": (
        ["Subhajit", "Rina", "Abhijit", "Moumita", "Sourav"],
        ["Das", "Mondal", "Ghosh", "Banerjee", "Saha"],
    ),
    "Odia": (["Ananya", "Bikram", "Smruti", "Debasis"], ["Patnaik", "Mohanty", "Sahoo", "Behera"]),
    "Assamese": (["Bikash", "Rupjyoti", "Pranami"], ["Gogoi", "Bora", "Kalita"]),
    "Malayalam": (
        ["Joseph", "Anu", "Rahul", "Reshma", "Shibu"],
        ["Thomas", "Nair", "Varghese", "Pillai"],
    ),
    "Urdu": (
        ["Mohammed", "Ayesha", "Salim", "Farhan", "Nazia"],
        ["Imran", "Khan", "Shaikh", "Ansari", "Qureshi"],
    ),
}

# education level -> invented institutions
INSTITUTIONS: dict[str, list[str]] = {
    "Below 10th": ["Govt. Primary School, {town}", "Zilla Parishad School, {town}"],
    "10th": [
        "Govt. High School, {town}",
        "Sri Saraswati Vidya Mandir, {town}",
        "St. Anne's High School, {town}",
    ],
    "12th": ["Govt. Junior College, {town}", "Model Higher Secondary School, {town}"],
    "ITI": ["Govt. ITI, {town}", "Vishwakarma Private ITI, {town}"],
    "Diploma": ["Govt. Polytechnic, {town}", "Institute of Hotel Management Studies, {town}"],
    "Graduate": [
        "Govt. Arts & Science College, {town}",
        "Sri Venkateshwara Degree College, {town}",
    ],
    "Postgraduate": ["Regional Institute of Commerce, {town}"],
}

JOBS: list[JobRequirement] = [
    JobRequirement(
        job_id="J1",
        role="housekeeping",
        title="Housekeeping Staff",
        openings=10,
        location="Bengaluru",
        min_years=1,
        languages_any=["Kannada", "Hindi"],
        preferred_skills=["room cleaning", "bathroom sanitation", "machine scrubbing"],
        max_salary_monthly_inr=18_000,
        max_notice_days=15,
        description="10 housekeeping staff, Bangalore, 1+ year experience, Kannada or Hindi.",
    ),
    JobRequirement(
        job_id="J2",
        role="front_desk",
        title="Receptionist",
        openings=2,
        location="Pune",
        min_years=2,
        languages_required=["English"],
        preferred_skills=["guest check-in", "phone handling", "MS Office"],
        max_salary_monthly_inr=25_000,
        max_notice_days=30,
        description=(
            "2 receptionists for a hotel in Pune, 2+ years front desk, good English, MS Office."
        ),
    ),
    JobRequirement(
        job_id="J3",
        role="security",
        title="Security Guard",
        openings=15,
        location="Hyderabad",
        min_years=1,
        languages_any=["Telugu", "Hindi"],
        preferred_skills=["access control", "patrolling", "CCTV monitoring"],
        max_salary_monthly_inr=20_000,
        max_notice_days=7,
        notice_stretch_days=7,  # "join within a week": one more week at most
        description=(
            "15 security guards for a tech park in Hyderabad, 1+ year, Telugu or Hindi, "
            "join within a week."
        ),
    ),
    JobRequirement(
        job_id="J4",
        role="delivery",
        title="Delivery Executive",
        openings=20,
        location="Chennai",
        min_years=0,
        languages_any=["Tamil"],
        required_certifications=["Two-Wheeler Driving Licence"],
        preferred_skills=["two-wheeler riding", "route navigation", "cash handling"],
        max_salary_monthly_inr=22_000,
        max_notice_days=7,
        notice_stretch_days=7,  # "join within a week": one more week at most
        description=(
            "20 delivery riders in Chennai, freshers OK, must speak Tamil and hold a "
            "2-wheeler licence."
        ),
    ),
    JobRequirement(
        job_id="J5",
        role="customer_support",
        title="Customer Support Executive",
        openings=8,
        location="Noida",
        min_years=1,
        languages_required=["English", "Hindi"],
        preferred_skills=["inbound calls", "complaint handling", "CRM software"],
        max_salary_monthly_inr=26_000,
        max_notice_days=30,
        description=(
            "8 customer support executives (voice) in Noida, 1+ year, fluent Hindi and English."
        ),
    ),
]
