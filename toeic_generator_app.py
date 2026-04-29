"""
TOEIC AI Question Generator (Streamlit)
========================================
streamlit run toeic_generator_app.py
"""

import streamlit as st
import json, time, random, re, os, base64, subprocess
from pathlib import Path
from io import BytesIO
from datetime import datetime

try:
    import requests
except ImportError:
    st.error("pip install requests")
    st.stop()

# .env loader
def load_dotenv():
    script_dir = Path(__file__).parent.resolve()
    for env_path in [script_dir / ".env", Path(".env")]:
        if env_path.exists():
            print(f"[ENV] Loading: {env_path}", flush=True)
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    os.environ[k] = v
                    print(f"[ENV]   {k} = {v[:6]}***", flush=True)
            return
    print("[ENV] No .env found", flush=True)

# Run init only once per process (Streamlit re-imports module on every rerun)
if not getattr(st, "_toeic_app_initialized", False):
    load_dotenv()
    print("\n" + "="*60, flush=True)
    print("  TOEIC Generator App Started", flush=True)
    print("="*60, flush=True)
    st._toeic_app_initialized = True

st.set_page_config(page_title="TOEIC Generator", page_icon="📝", layout="wide")

# Fragment decorator for tab isolation (Streamlit 1.33+)
try:
    _fragment = st.fragment
except AttributeError:
    _fragment = lambda f: f  # no-op fallback for older Streamlit


# ══════════════════════════════════════
# Model Definitions
# ══════════════════════════════════════
MODEL_OPTIONS = {
    "gemma3:12b (12B local GPU)": {"engine":"ollama","model":"gemma3:12b"},
    "gemma4:e4b (4B local GPU)": {"engine":"ollama","model":"gemma4:e4b"},
    "gemma4:26b (MoE 26B local)": {"engine":"ollama","model":"gemma4:26b"},
    "gemma3:4b (4B lightweight)": {"engine":"ollama","model":"gemma3:4b"},
    "gemini-2.5-flash-lite (API fastest)": {"engine":"gemini","model":"gemini-2.5-flash-lite"},
    "gemini-2.5-flash (API balanced)": {"engine":"gemini","model":"gemini-2.5-flash"},
    "gemini-3-flash (API best value)": {"engine":"gemini","model":"gemini-3-flash-preview"},
    "gemini-2.5-pro (API premium)": {"engine":"gemini","model":"gemini-2.5-pro"},
}

GEMINI_THINKING = {"gemini-2.5-flash-lite":0,"gemini-2.5-flash":0,"gemini-3-flash-preview":0,"gemini-2.5-pro":1024}

# 503フォールバックチェーン: 3-Flash → 2.5-Flash → Flash-Lite → ローカル gemma3
GEMINI_FALLBACK_CHAIN = [
    {"engine":"gemini","model":"gemini-3-flash-preview",  "label":"3-Flash"},
    {"engine":"gemini","model":"gemini-2.5-flash",        "label":"2.5-Flash"},
    {"engine":"gemini","model":"gemini-2.5-flash-lite",   "label":"Flash-Lite"},
    {"engine":"ollama","model":"gemma3:12b",              "label":"gemma3:12b (local)"},
]

# パート別の推奨モデル（3-Flash = 高性能+低コスト）
PART_DEFAULT_MODEL = {
    "part1": "gemini-3-flash (API best value)",
    "part2": "gemini-3-flash (API best value)",
    "part3": "gemini-3-flash (API best value)",
    "part3_3p": "gemini-3-flash (API best value)",
    "part4": "gemini-3-flash (API best value)",
    "part5": "gemini-3-flash (API best value)",
    "part6": "gemini-3-flash (API best value)",
    "part7": "gemini-3-flash (API best value)",
    "part7s": "gemini-3-flash (API best value)",
    "part7d": "gemini-3-flash (API best value)",
    "part7t": "gemini-3-flash (API best value)",
}

# Recommended generation level per part (based on part structural difficulty limits)
PART_REC_LEVEL = {
    "part1": "intermediate",   # Photo: structural limit ~550
    "part2": "intermediate",   # Q&A: structural limit ~650
    "part3": "advanced",       # Conversation: can be 450-800
    "part3_3p": "advanced",
    "part4": "advanced",       # Talk: can be 450-800
    "part5": "advanced",       # Grammar: full range 400-950
    "part6": "advanced",       # Text completion: 550-850
    "part7": "advanced",       # Reading: full range 450-900
    "part7s": "advanced",
    "part7d": "advanced",
    "part7t": "advanced",
}

# ══════════════════════════════════════
# Type Pools (151 types)
# ══════════════════════════════════════
TYPES = {"part1":[{"type": "office_desk", "desc": "Person working at desk — typing, reading, organizing"}, {"type": "meeting_conference", "desc": "People in conference room — presenting, discussing"}, {"type": "reception_lobby", "desc": "Reception area — receptionist greeting, visitor signing in"}, {"type": "copy_room", "desc": "Copy/print room — person using copier, sorting papers"}, {"type": "break_room", "desc": "Break room — people having coffee, using microwave, vending machine"}, {"type": "server_room", "desc": "Server/IT room — technician checking equipment, cables"}, {"type": "mailroom_delivery", "desc": "Mailroom — sorting packages, loading cart, delivery person"}, {"type": "filing_cabinet", "desc": "Storage/filing area — person opening cabinet, organizing folders"}, {"type": "whiteboard_planning", "desc": "Planning session — person writing on whiteboard, sticky notes"}, {"type": "elevator_hallway", "desc": "Elevator/hallway — people waiting, walking, carrying items"}, {"type": "staircase_hallway", "desc": "Building interior — staircase, hallway, elevator area"}, {"type": "rooftop_terrace", "desc": "Office rooftop/terrace — outdoor seating, city view"}, {"type": "retail_shopping", "desc": "Store — customer browsing, cashier, trying on clothes"}, {"type": "supermarket_grocery", "desc": "Grocery store — shopping cart, produce aisle, checkout"}, {"type": "pharmacy_counter", "desc": "Pharmacy — pharmacist behind counter, customer picking up prescription"}, {"type": "bookstore_browsing", "desc": "Bookstore — person reading, shelves of books, display table"}, {"type": "electronics_store", "desc": "Electronics store — customer looking at laptops, TVs on display"}, {"type": "jewelry_boutique", "desc": "Jewelry/boutique shop — display case, customer being shown items"}, {"type": "restaurant_cafe", "desc": "Café or restaurant — waiter serving, people dining, barista"}, {"type": "kitchen_cooking", "desc": "Kitchen — chef cooking, preparing food, cutting vegetables"}, {"type": "food_truck_vendor", "desc": "Street food — food cart, vendor serving, customers queueing"}, {"type": "bakery_counter", "desc": "Bakery — bread display, baker arranging pastries, customer pointing"}, {"type": "buffet_catering", "desc": "Buffet/catering setup — trays of food, people serving themselves"}, {"type": "outdoor_dining", "desc": "Outdoor restaurant patio — tables with umbrellas, server carrying tray"}, {"type": "train_station", "desc": "Train platform — passengers waiting, boarding, luggage"}, {"type": "airport_terminal", "desc": "Airport — check-in counter, gate, luggage carousel"}, {"type": "bus_stop", "desc": "Bus stop — people waiting, bus arriving, person boarding"}, {"type": "parking_lot", "desc": "Parking area — cars lined up, person loading trunk"}, {"type": "taxi_rideshare", "desc": "Taxi/car — passenger getting in/out, driver opening door"}, {"type": "ferry_dock", "desc": "Ferry terminal — passengers boarding, cars driving onto ferry"}, {"type": "construction_site", "desc": "Construction — scaffolding, crane, workers with helmets"}, {"type": "warehouse_factory", "desc": "Warehouse or factory — forklift, conveyor, stacking boxes"}, {"type": "workshop_tools", "desc": "Workshop — carpenter, mechanic, tools on workbench"}, {"type": "loading_dock", "desc": "Loading dock — truck backed up, workers unloading pallets"}, {"type": "lab_research", "desc": "Laboratory — person in lab coat, microscope, test tubes"}, {"type": "assembly_line", "desc": "Assembly line — workers at stations, products on conveyor"}, {"type": "park_bench", "desc": "Park — person reading on bench, walking dog, jogger"}, {"type": "street_crosswalk", "desc": "City street — pedestrians crossing, traffic, bus stop"}, {"type": "garden_yard", "desc": "Garden — planting flowers, mowing lawn, raking leaves"}, {"type": "bridge_river", "desc": "Bridge or river — people crossing bridge, riverbank view"}, {"type": "marina_dock", "desc": "Waterfront — boats docked, fishing, loading cargo"}, {"type": "beach_waterfront", "desc": "Beach — people walking, umbrellas, lifeguard tower"}, {"type": "hiking_trail", "desc": "Hiking trail — hikers with backpacks, trail sign, mountain view"}, {"type": "fountain_plaza", "desc": "City plaza — fountain, people sitting on steps, street performer"}, {"type": "medical_clinic", "desc": "Clinic or hospital — doctor, nurse, patient, waiting room"}, {"type": "library_bookstore", "desc": "Library — shelves of books, person reading, studying"}, {"type": "museum_gallery", "desc": "Museum or art gallery — paintings on wall, visitors looking"}, {"type": "hotel_reception", "desc": "Hotel — guest checking in, bellhop, lobby seating"}, {"type": "gym_sports", "desc": "Gym or sports — exercising, swimming pool, tennis court"}, {"type": "classroom_lecture", "desc": "Classroom — teacher at board, students taking notes"}, {"type": "post_office", "desc": "Post office — customer at counter, weighing package, stamp display"}, {"type": "laundromat", "desc": "Laundromat — person loading washing machine, folding clothes"}, {"type": "moving_packing", "desc": "Moving — carrying boxes, loading truck, wrapping furniture"}, {"type": "cleaning_maintenance", "desc": "Cleaning — mopping floor, wiping window, vacuum"}, {"type": "bicycle_cyclist", "desc": "Bicycle scene — person riding, bikes parked in rack"}, {"type": "outdoor_market", "desc": "Open-air market — vendors, produce stands, shoppers"}, {"type": "painting_renovation", "desc": "Painting/renovation — person on ladder, painting wall, drop cloth"}, {"type": "gardening_landscaping", "desc": "Landscaping — trimming hedges, planting tree, wheelbarrow"}, {"type": "window_shopping", "desc": "Window shopping — person looking at storefront display, mannequins"}, {"type": "empty_room_furniture", "desc": "Empty room (NO people) — chairs, tables, equipment arranged"}],"part2":[{"type": "wh_what_thing", "desc": "What + thing/object. 'What kind of printer do we need?'"}, {"type": "wh_what_action", "desc": "What + action/plan. 'What should we do about the leak?'"}, {"type": "wh_what_time", "desc": "What time/day. 'What time does the store close?'"}, {"type": "wh_which", "desc": "Which + choice. 'Which floor is HR on?' / 'Which report?'"}, {"type": "wh_where_place", "desc": "Where + location. 'Where is the supply room?'"}, {"type": "wh_where_direction", "desc": "Where + direction. 'Where should I put these boxes?'"}, {"type": "wh_when_future", "desc": "When + future. 'When will the shipment arrive?'"}, {"type": "wh_when_past", "desc": "When + past. 'When did you last speak with the client?'"}, {"type": "wh_who_person", "desc": "Who + person. 'Who is in charge of the project?'"}, {"type": "wh_whose", "desc": "Whose + possession. 'Whose jacket is this?'"}, {"type": "wh_why", "desc": "Why question. 'Why was the meeting postponed?'"}, {"type": "wh_how_manner", "desc": "How + manner. 'How did the presentation go?'"}, {"type": "wh_how_long", "desc": "How long/far. 'How long will the renovation take?'"}, {"type": "wh_how_many", "desc": "How many/much. 'How many copies do you need?'"}, {"type": "wh_how_often", "desc": "How often/soon. 'How often does this bus run?'"}, {"type": "yesno_do", "desc": "Do/Does/Did question. 'Do you have the agenda?'"}, {"type": "yesno_be", "desc": "Is/Are/Was question. 'Is the report ready?'"}, {"type": "yesno_have", "desc": "Have/Has question. 'Have you finished the proposal?'"}, {"type": "yesno_can", "desc": "Can/Could question. 'Can you work Saturday?'"}, {"type": "yesno_will", "desc": "Will/Would question. 'Will the store be open tomorrow?'"}, {"type": "yesno_should", "desc": "Should question. 'Should we order more supplies?'"}, {"type": "negative_isnt", "desc": "Negative isn't/aren't. 'Isn't the deadline tomorrow?'"}, {"type": "negative_didnt", "desc": "Negative didn't/don't. 'Didn't you attend the workshop?'"}, {"type": "tag", "desc": "Tag question. 'The report is due Friday, isn't it?'"}, {"type": "choice", "desc": "Choice A or B. 'Morning or afternoon session?'"}, {"type": "embedded", "desc": "Embedded question. 'Do you know where the supply room is?'"}, {"type": "statement_problem", "desc": "Statement: problem. 'The printer seems to be jammed.'"}, {"type": "statement_plan", "desc": "Statement: plan/fact. 'I'm visiting the new office tomorrow.'"}, {"type": "statement_opinion", "desc": "Statement: opinion. 'That was an excellent presentation.'"}, {"type": "statement_news", "desc": "Statement: news. 'The CEO just announced a merger.'"}, {"type": "request", "desc": "Request. 'Could you help me carry these boxes?'"}, {"type": "suggestion_offer", "desc": "Suggestion/offer. 'Why don't we take a break?'"}],"part3":[{"type": "office_equipment", "desc": "Equipment issue — printer jam, computer crash, phone system"}, {"type": "schedule_change", "desc": "Meeting reschedule, deadline extension, shift swap"}, {"type": "project_discussion", "desc": "Project progress update, milestone review, resource allocation"}, {"type": "coworker_favor", "desc": "Colleague asking for help, covering shift, borrowing supplies"}, {"type": "promotion_transfer", "desc": "Promotion discussion, department transfer, new role"}, {"type": "office_relocation", "desc": "Office move, desk reassignment, floor renovation"}, {"type": "client_negotiation", "desc": "Client call, contract terms, pricing discussion"}, {"type": "complaint_resolution", "desc": "Customer complaint handling, service recovery, refund"}, {"type": "performance_review", "desc": "Performance evaluation, goal setting, feedback session"}, {"type": "new_employee", "desc": "New hire orientation, introducing to team, explaining procedures"}, {"type": "salary_benefits", "desc": "Salary discussion, benefits enrollment, insurance question"}, {"type": "remote_work", "desc": "Work from home arrangement, video call setup, schedule flexibility"}, {"type": "restaurant_order", "desc": "Restaurant reservation, order, dietary restriction, bill"}, {"type": "hotel_checkin", "desc": "Hotel check-in, room upgrade, amenities, late checkout"}, {"type": "retail_purchase", "desc": "Store purchase, exchange, size inquiry, loyalty card"}, {"type": "bank_finance", "desc": "Bank transaction, loan inquiry, account opening, wire transfer"}, {"type": "repair_maintenance", "desc": "Car repair quote, appliance fix, warranty claim"}, {"type": "medical_pharmacy", "desc": "Doctor appointment, prescription refill, insurance claim"}, {"type": "gym_membership", "desc": "Gym membership signup, class schedule, personal trainer"}, {"type": "dry_cleaner", "desc": "Dry cleaning pickup, stain removal, rush order"}, {"type": "airport_travel", "desc": "Flight delay, booking change, seat upgrade, baggage"}, {"type": "commute_transport", "desc": "Parking issue, bus route change, carpool arrangement"}, {"type": "business_trip", "desc": "Travel planning, hotel booking, itinerary change, expense"}, {"type": "car_rental", "desc": "Car rental reservation, insurance option, return location"}, {"type": "tour_booking", "desc": "Tour reservation, group discount, cancellation policy"}, {"type": "moving_relocation", "desc": "Moving company quote, packing service, delivery date"}, {"type": "hiring_interview", "desc": "Job interview feedback, candidate comparison, reference check"}, {"type": "training_workshop", "desc": "Training enrollment, certification, online vs in-person"}, {"type": "event_planning", "desc": "Conference setup, venue selection, catering, speaker"}, {"type": "marketing_campaign", "desc": "Ad campaign planning, social media strategy, budget"}, {"type": "product_development", "desc": "Product feature discussion, testing feedback, launch timeline"}, {"type": "vendor_selection", "desc": "Comparing suppliers, price negotiation, delivery terms"}, {"type": "legal_contract", "desc": "Contract review, terms clarification, signing deadline"}, {"type": "tech_support", "desc": "Software issue, system update, password reset, network"}, {"type": "graphic_schedule", "desc": "GRAPHIC: schedule/timetable"}, {"type": "graphic_price_list", "desc": "GRAPHIC: price list/menu"}, {"type": "graphic_order_form", "desc": "GRAPHIC: order form/invoice"}, {"type": "graphic_floor_map", "desc": "GRAPHIC: floor map/seating chart"}],"part3_3p":[{"type": "team_meeting", "desc": "3 colleagues discussing project, deadline, or task assignment"}, {"type": "office_move", "desc": "3 people coordinating office relocation or desk arrangement"}, {"type": "event_coordination", "desc": "3 people planning a company event, party, or conference"}, {"type": "client_presentation", "desc": "3 colleagues preparing for or debriefing a client meeting"}, {"type": "hiring_decision", "desc": "3 people discussing job candidates or interview results"}, {"type": "budget_review", "desc": "3 people reviewing department budget or expense approval"}, {"type": "travel_arrangement", "desc": "3 colleagues arranging a group business trip"}, {"type": "training_feedback", "desc": "3 people discussing training session content or schedule"}, {"type": "lunch_plans", "desc": "3 coworkers deciding where to eat or planning lunch outing"}, {"type": "problem_solving", "desc": "3 people troubleshooting a technical or logistics issue"}, {"type": "product_launch", "desc": "3 people finalizing product launch — marketing, timing, pricing"}, {"type": "department_merger", "desc": "3 people from different teams discussing merger or reorganization"}, {"type": "safety_inspection", "desc": "3 people reviewing safety audit findings or compliance steps"}, {"type": "customer_feedback", "desc": "3 people analyzing survey results or customer complaints"}, {"type": "volunteer_planning", "desc": "3 people organizing a charity or community volunteer event"}],"part4":[{"type": "voicemail", "desc": "Voicemail about scheduling, delivery, or callback"}, {"type": "announcement_office", "desc": "Office announcement: policy change, visitor, maintenance"}, {"type": "announcement_store", "desc": "Store/venue: sale, closing time, special event"}, {"type": "announcement_transport", "desc": "Airport/station: delay, gate change, boarding call"}, {"type": "announcement_building", "desc": "Building: fire drill, elevator maintenance, parking"}, {"type": "meeting_intro", "desc": "Meeting opening: agenda overview, speaker introduction"}, {"type": "tour_guide", "desc": "Tour guide: museum, factory, campus, city"}, {"type": "news_business", "desc": "Business news: earnings, merger, market trends"}, {"type": "news_weather", "desc": "Weather forecast: weekend outlook, storm warning"}, {"type": "news_traffic", "desc": "Traffic update: road closure, accident, alternate route"}, {"type": "training_safety", "desc": "Safety training: emergency procedure, equipment, fire exit"}, {"type": "training_software", "desc": "Software tutorial: new system rollout, feature walkthrough"}, {"type": "training_onboarding", "desc": "New employee orientation: company policy, benefits"}, {"type": "award_ceremony", "desc": "Award speech: employee of month, retirement, achievement"}, {"type": "product_launch", "desc": "Product presentation: features, pricing, availability"}, {"type": "conference_keynote", "desc": "Conference keynote: industry trends, research findings"}, {"type": "radio_advertisement", "desc": "Radio ad: product promotion, grand opening, limited offer"}, {"type": "radio_interview", "desc": "Radio interview: expert opinion, book promotion"}, {"type": "helpline_recording", "desc": "Automated phone: menu options, hours, hold message"}, {"type": "workshop_promo", "desc": "Workshop/seminar: registration, schedule, benefits"}, {"type": "construction_notice", "desc": "Construction/renovation: timeline, detours, noise"}, {"type": "charity_volunteer", "desc": "Charity event or volunteer recruitment appeal"}, {"type": "company_update", "desc": "CEO/manager update: results, restructuring, new hire"}, {"type": "library_announcement", "desc": "Library: new hours, book sale, author event"}, {"type": "gym_class", "desc": "Gym/fitness: schedule change, new instructor, class intro"}, {"type": "apartment_notice", "desc": "Apartment: maintenance, rent change, amenity update"}, {"type": "graduation_speech", "desc": "Graduation address: achievement, future advice"}, {"type": "museum_audio_guide", "desc": "Museum audio guide: exhibit description, artist info"}, {"type": "graphic_schedule", "desc": "GRAPHIC: schedule/agenda"}, {"type": "graphic_price", "desc": "GRAPHIC: price list/rate card"}, {"type": "graphic_map", "desc": "GRAPHIC: map/floor plan"}, {"type": "graphic_chart", "desc": "GRAPHIC: bar/pie chart with data"}],"part5":[{"type": "word_form_noun_verb", "desc": "Word form: noun vs verb (decision/decide)"}, {"type": "word_form_adj_adv", "desc": "Word form: adj vs adverb (careful/carefully)"}, {"type": "word_form_noun_adj", "desc": "Word form: noun vs adj (confidence/confident)"}, {"type": "word_form_verb_adj", "desc": "Word form: verb vs adj (satisfy/satisfactory)"}, {"type": "word_form_negative", "desc": "Word form: negative prefix (un-/in-/dis-)"}, {"type": "participle_adj", "desc": "Participle: -ing vs -ed (interesting/interested)"}, {"type": "compound_noun", "desc": "Compound modifier (cost-effective, time-consuming)"}, {"type": "word_form_er_ee", "desc": "Word form: -er/-or vs -ee (employer/employee)"}, {"type": "verb_tense_present", "desc": "Verb tense: present simple/continuous"}, {"type": "verb_tense_past", "desc": "Verb tense: past simple/perfect"}, {"type": "verb_tense_future", "desc": "Verb tense: future (will/going to)"}, {"type": "passive_voice", "desc": "Active vs passive (was completed, being reviewed)"}, {"type": "subject_verb_agreement", "desc": "Subject-verb agreement (each IS, a number ARE)"}, {"type": "preposition_basic", "desc": "Basic preposition (in/on/at/for/to/by/with)"}, {"type": "preposition_idiom", "desc": "Prepositional idiom (responsible FOR, comply WITH)"}, {"type": "conjunction_contrast", "desc": "Contrast (although, whereas, despite, nevertheless)"}, {"type": "conjunction_cause", "desc": "Cause/result (because, therefore, due to)"}, {"type": "conjunction_condition", "desc": "Condition (if, unless, provided that, as long as)"}, {"type": "preposition_vs_conjunction", "desc": "Prep vs conj (during/while, because of/because)"}, {"type": "pronoun_relative", "desc": "Relative pronoun (who/which/that/whose/whom)"}, {"type": "pronoun_reflexive", "desc": "Reflexive/possessive (themselves, its, one another)"}, {"type": "pronoun_indefinite", "desc": "Indefinite (anyone, each, either, neither, several)"}, {"type": "vocab_business", "desc": "Business vocab (revenue, acquisition, subsidiary)"}, {"type": "vocab_office", "desc": "Office vocab (submit, forward, distribute, reschedule)"}, {"type": "vocab_hr", "desc": "HR vocab (recruit, promote, resign, compensation)"}, {"type": "vocab_finance", "desc": "Finance vocab (invoice, reimburse, allocate, budget)"}, {"type": "vocab_marketing", "desc": "Marketing vocab (launch, campaign, survey, demographic)"}, {"type": "vocab_logistics", "desc": "Logistics vocab (shipment, warehouse, inventory, vendor)"}, {"type": "vocab_legal", "desc": "Legal vocab (comply, regulation, authorize, mandatory)"}, {"type": "vocab_collocation", "desc": "Collocation (make a reservation, meet a deadline)"}, {"type": "vocab_phrasal", "desc": "Phrasal verb (carry out, look into, set up, turn down)"}, {"type": "vocab_context", "desc": "Context vocab (appropriate, considerable, thorough)"}, {"type": "gerund_infinitive", "desc": "Gerund vs infinitive (enjoy doing, plan to do)"}, {"type": "comparative_superlative", "desc": "Comparative/superlative (more efficient, the most)"}],"part6":[{"type": "email_apology", "desc": "Apology for delay, error, or service failure"}, {"type": "email_announcement", "desc": "Company announcement: new policy, system change"}, {"type": "email_invitation", "desc": "Invitation to event, meeting, or celebration"}, {"type": "email_request", "desc": "Request for information, approval, or assistance"}, {"type": "email_followup", "desc": "Follow-up after meeting, interview, or inquiry"}, {"type": "email_introduction", "desc": "Introducing new colleague, vendor, or service provider"}, {"type": "email_confirmation", "desc": "Confirming reservation, order, or appointment"}, {"type": "email_thankyou", "desc": "Thank-you for purchase, attendance, or cooperation"}, {"type": "formal_letter_offer", "desc": "Job offer or contract acceptance letter"}, {"type": "formal_letter_complaint", "desc": "Complaint about product, service, or billing"}, {"type": "formal_letter_recommendation", "desc": "Recommendation letter for employee or student"}, {"type": "formal_letter_resignation", "desc": "Resignation notice or farewell letter"}, {"type": "memo_policy", "desc": "Internal memo: policy or procedure update"}, {"type": "memo_event", "desc": "Internal memo: company event, team building"}, {"type": "memo_budget", "desc": "Internal memo: budget update, expense guidelines"}, {"type": "memo_staffing", "desc": "Internal memo: hiring freeze, overtime, schedule change"}, {"type": "notice_closure", "desc": "Temporary closure or renovation notice"}, {"type": "notice_rules", "desc": "Parking rules, building regulations, safety guidelines"}, {"type": "notice_construction", "desc": "Construction/maintenance notice with timeline"}, {"type": "notice_recall", "desc": "Product recall or safety alert notice"}, {"type": "advertisement_product", "desc": "Product ad with features and promotional offer"}, {"type": "advertisement_recruitment", "desc": "Job posting with qualifications and benefits"}, {"type": "newsletter_article", "desc": "Newsletter excerpt: company news, industry update"}, {"type": "review_response", "desc": "Business review + management response"}, {"type": "instruction_manual", "desc": "User guide, setup instructions, or FAQ excerpt"}, {"type": "press_release", "desc": "Company press release: partnership, expansion, milestone"}],"part7s":[{"type": "email_internal", "desc": "Internal company email"}, {"type": "email_external", "desc": "External business email (client, vendor)"}, {"type": "letter_formal", "desc": "Formal business letter"}, {"type": "online_chat", "desc": "Online customer service chat"}, {"type": "text_message", "desc": "Text/instant message conversation"}, {"type": "memo_internal", "desc": "Internal office memo"}, {"type": "article_business", "desc": "Business magazine/newspaper article"}, {"type": "article_community", "desc": "Community newsletter or local news"}, {"type": "blog_post", "desc": "Company blog post or industry commentary"}, {"type": "press_release", "desc": "Company press release announcement"}, {"type": "advertisement_job", "desc": "Job posting/recruitment ad"}, {"type": "advertisement_product", "desc": "Product or service advertisement"}, {"type": "advertisement_event", "desc": "Event promotion or workshop announcement"}, {"type": "coupon_promotion", "desc": "Coupon, discount code, or loyalty program"}, {"type": "notice_building", "desc": "Building/facility notice"}, {"type": "notice_policy", "desc": "Policy update or regulation change"}, {"type": "form_survey", "desc": "Application form, survey, or questionnaire"}, {"type": "schedule_timetable", "desc": "Schedule, timetable, or agenda"}, {"type": "invoice_receipt", "desc": "Invoice, receipt, or billing statement"}, {"type": "webpage_listing", "desc": "Website product listing or service page"}, {"type": "review_response", "desc": "Customer review + business response"}, {"type": "meeting_minutes", "desc": "Meeting minutes or action items summary"}, {"type": "travel_itinerary", "desc": "Travel itinerary or booking confirmation"}, {"type": "directory_listing", "desc": "Staff directory or contact page"}, {"type": "graphic_invoice", "desc": "GRAPHIC: invoice with table"}, {"type": "graphic_survey_results", "desc": "GRAPHIC: survey results chart"}, {"type": "graphic_comparison_chart", "desc": "GRAPHIC: comparison table"}, {"type": "graphic_schedule_event", "desc": "GRAPHIC: event schedule"}],"part7d":[{"type": "job_posting_application", "desc": "Doc1: Job ad. Doc2: Cover letter/application"}, {"type": "event_flyer_feedback", "desc": "Doc1: Event flyer. Doc2: Attendee feedback"}, {"type": "product_ad_review", "desc": "Doc1: Product ad. Doc2: Customer review"}, {"type": "memo_reply", "desc": "Doc1: Internal memo. Doc2: Reply email"}, {"type": "hotel_booking_change", "desc": "Doc1: Hotel confirmation. Doc2: Change request"}, {"type": "invoice_dispute", "desc": "Doc1: Invoice. Doc2: Dispute email"}, {"type": "conference_schedule_chat", "desc": "Doc1: Conference schedule. Doc2: Attendee chat"}, {"type": "coupon_receipt", "desc": "Doc1: Coupon. Doc2: Purchase receipt"}, {"type": "travel_itinerary_change", "desc": "Doc1: Travel itinerary. Doc2: Change request"}, {"type": "newsletter_reader_letter", "desc": "Doc1: Newsletter. Doc2: Reader response"}, {"type": "pricelist_inquiry", "desc": "Doc1: Price list. Doc2: Customer inquiry"}, {"type": "policy_question", "desc": "Doc1: Policy document. Doc2: Employee question"}, {"type": "lease_agreement_complaint", "desc": "Doc1: Lease agreement. Doc2: Tenant complaint"}, {"type": "training_enrollment", "desc": "Doc1: Training description. Doc2: Enrollment email"}, {"type": "warranty_claim", "desc": "Doc1: Warranty terms. Doc2: Claim email"}, {"type": "survey_results_memo", "desc": "Doc1: Survey results. Doc2: Action memo"}, {"type": "menu_catering_order", "desc": "Doc1: Catering menu. Doc2: Order confirmation"}, {"type": "resume_interview", "desc": "Doc1: Resume. Doc2: Interview invitation"}, {"type": "donation_thankyou", "desc": "Doc1: Donation appeal. Doc2: Thank-you letter"}, {"type": "insurance_claim", "desc": "Doc1: Insurance policy. Doc2: Claim form"}],"part7t":[{"type": "job_ad_resume_email", "desc": "Job ad + Resume + Interview email"}, {"type": "hotel_confirm_review_reply", "desc": "Hotel confirmation + Review + Manager reply"}, {"type": "event_schedule_email_feedback", "desc": "Event schedule + Organizer email + Feedback"}, {"type": "product_catalog_order_complaint", "desc": "Catalog + Order + Complaint email"}, {"type": "policy_memo_faq_email", "desc": "Policy + FAQ + Employee question"}, {"type": "travel_itinerary_change_expense", "desc": "Itinerary + Change request + Expense report"}, {"type": "ad_coupon_receipt", "desc": "Store ad + Coupon + Receipt"}, {"type": "newsletter_article_reader", "desc": "Newsletter + Article + Reader comment"}, {"type": "training_schedule_eval_cert", "desc": "Training schedule + Evaluation + Certificate"}, {"type": "restaurant_menu_review_reply", "desc": "Menu + Review + Owner reply"}, {"type": "apartment_lease_complaint_response", "desc": "Lease + Complaint + Management response"}, {"type": "conference_reg_invoice_receipt", "desc": "Brochure + Registration + Invoice"}, {"type": "insurance_claim_assessment", "desc": "Policy + Claim form + Assessment letter"}, {"type": "warranty_repair_feedback", "desc": "Warranty + Repair request + Service feedback"}, {"type": "scholarship_application_result", "desc": "Announcement + Application + Award letter"}, {"type": "vendor_proposal_contract", "desc": "RFP + Vendor proposal + Contract award"}, {"type": "renovation_plan_update_complaint", "desc": "Renovation plan + Progress update + Complaint"}, {"type": "membership_renewal_benefits", "desc": "Membership overview + Renewal + Benefits update"}]}

LEVEL_GUIDES = {
    "beginner": """DIFFICULTY: EASY (questions a ~600 scorer can answer — about 60% correct rate)
VOCABULARY: basic everyday + entry-level business words (meeting, office, schedule, delivery, appointment, department, available, confirm)
CORRECT ANSWER: direct and clear — answer matches keywords in the question or text with minimal paraphrasing
DISTRACTORS: use common TOEIC traps but keep them identifiable: repeated words from the question in wrong context, similar-sounding words, wrong tense
  Example Part 2: "Where is the meeting room?" → "It's on the second floor." (direct WH-answer)
  Example Part 5: "The packages will be ------- tomorrow morning." → delivered (basic passive future — test word form: deliver/delivery/delivering/delivered)
  Example Part 7: "What is the purpose of this email?" (answer directly stated in the first sentence)
GRAMMAR TESTED: present/past/future simple, basic passive (is delivered), simple word forms (noun vs verb vs adjective), basic prepositions (in/on/at/by)
IMPORTANT: Audio speed, conversation length, and document length are IDENTICAL to all levels. Only vocabulary complexity, answer directness, and distractor subtlety differ.""",
    "intermediate": """DIFFICULTY: MODERATE (questions a 600-800 scorer can answer — about 65-80% correct rate)
VOCABULARY: established business terms (negotiate, implement, quarterly, renovation, mandatory, prospective, reimburse, accommodate)
CORRECT ANSWER: often paraphrased from the original text — requires understanding meaning, not just matching keywords. Some indirect responses in Part 2.
DISTRACTORS: plausible and related to the topic — require careful reading/listening to eliminate. Include partial-truth traps (choice mentions something true but doesn't answer the question).
  Example Part 2: "Has the quarterly report been submitted yet?" → "Ms. Park is still finalizing the data." (indirect — implies no, without saying no)
  Example Part 5: "All employees are required to ------- the safety training by the end of this month." → complete (vocabulary in context — test: complete/completion/completely/completed)
  Example Part 7: "What is suggested about the company?" / "What will Mr. Lee most likely do next?" (inference from context, not directly stated)
GRAMMAR TESTED: present perfect vs past simple, passive voice variations, conditionals, relative clauses, gerund vs infinitive, subject-verb agreement with complex subjects
IMPORTANT: Audio speed, conversation length, and document length are IDENTICAL to all levels. Only vocabulary complexity, answer directness, and distractor subtlety differ.""",
    "advanced": """DIFFICULTY: HARD (questions a 800+ scorer can answer — requires 80%+ correct rate)
VOCABULARY: For Part 5/6/7 — use sophisticated business vocabulary from diverse fields: procurement, compliance, amendment, forthcoming, expedite, remuneration, discretionary, contingent upon, arbitration, disclosure, depreciation, subsidiary, litigation, scalability. For Part 2 — use NORMAL workplace vocabulary (the difficulty comes from indirect responses, NOT from exotic words). For Part 3/4 — use natural conversational English with some business terms.
CORRECT ANSWER: heavily paraphrased, indirect, or requires synthesizing information from multiple parts of the text or multiple documents. In Part 2, use negative questions with indirect responses.
DISTRACTORS: use common TOEIC traps but keep them identifiable: repeated words from the question in wrong context, similar-sounding words, information that is true but doesn't answer the specific question. For Part 2: distractors should be CLEARLY WRONG for a specific reason (topic shift, word trap), NOT near-correct. For Part 5/7: distractors can be more subtle (near-synonyms, one-word difference).
  Example Part 2: "Hasn't the new schedule been posted yet?" → "I think Karen is still working on it." (negative question + normal vocabulary + indirect response)
  Example Part 5: "The merger, ------- initially opposed by a majority of shareholders, ultimately received regulatory approval." → although/despite/whereas/nevertheless (conjunction in complex sentence with interrupting clause)
  Example Part 7: "What can be inferred from both documents?" / "In the second email, what does the phrase 'moving forward' most likely mean?" / "What is NOT indicated about the renovation project?" (cross-reference, vocabulary-in-context, NOT questions)
GRAMMAR TESTED: subjunctive (recommend that he attend), inversion (Not only did...), participle clauses, advanced collocations (comply with, in accordance with), parallel structure in formal context
IMPORTANT: Audio speed, conversation length, and document length are IDENTICAL to all levels. Only vocabulary complexity, answer directness, and distractor subtlety differ.""",
}

# ══════════════════════════════════════
# Prompt Builder
# ══════════════════════════════════════
JA = '\nCRITICAL: ALL "explanation_ja" and "translation_ja" fields MUST be written in 日本語 (Japanese). NEVER write explanations in English.'
EN_EXPL = '\nENGLISH EXPLANATION: For each question, ALSO include an "explanation_en" field — a SHORT 1-2 sentence explanation in English (max 25 words) summarizing WHY the correct answer is right. Concise, clear, spoken-English style. This is separate from explanation_ja (which stays detailed in Japanese).'
BLANK_RULE = '\nCRITICAL BLANK RULE — The sentence MUST contain EXACTLY ONE blank written as seven hyphens: -------. The blank replaces the word being tested. A sentence WITHOUT ------- is INVALID and will be REJECTED. WRONG: "The company will launch its product." RIGHT: "The company will ------- its product."'
BLANK_RULE6 = '\nCRITICAL: The text MUST contain exactly 4 blanks written as (1)------- (2)------- (3)------- (4)-------. Without these blanks the output is INVALID.'
ANSWER_POSITION = '\nANSWER POSITION RULE — CRITICAL: The correct answer MUST be RANDOMLY distributed across positions. Do NOT always put the correct answer at (A)/index 0. Vary the "correct" field: use 0, 1, 2, 3 with roughly equal frequency. If generating multiple questions, NEVER make them all the same position.'
CHOICE_RULE = '\nCHOICE COUNT RULE — STRICT: Every question MUST have EXACTLY 4 choices labeled (A), (B), (C), (D). NEVER include (E) or more. Part 2 MUST have EXACTLY 3 choices (A), (B), (C). This is a hard TOEIC format requirement — violating it makes the output INVALID.'
VOCAB_RULE = '\nVOCABULARY EXTRACTION: Include a "vocab" array with exactly 3 key words/phrases. For each: "word" (English — always BASE/DICTIONARY form), "pos" (part of speech: "noun","verb","adjective","adverb","phrase"), "ja" (Japanese meaning), "example" (short example sentence), "level" (word difficulty 5-scale: "A1" = basic ~400 level like go/have/meeting/office, "A2" = elementary ~500-600 like schedule/confirm/delay/available, "B1" = intermediate 600-700 like negotiate/implement/mandatory/comply, "B2" = upper-intermediate 750-850 like procurement/reimburse/discretionary/contingent, "C" = advanced 900+ like arbitration/adjudicate/commensurate/promulgate). IMPORTANT: "level" reflects the WORD ITSELF difficulty, NOT the question difficulty. A basic word like "schedule" is always "A2" even in an advanced question. Choose words KEY to solving the question.'
PHRASE_RULE = '\nPHRASE REQUIREMENT: At least ONE of the 3 vocab items MUST be an IDIOMATIC phrase/collocation (pos:"phrase") — a multi-word expression whose meaning goes beyond its individual words. GOOD: "in compliance with", "prior to", "fill out", "on behalf of", "look forward to", "set up", "comply with". BAD: "company policy", "budget report", "new employee" (these are compound nouns, NOT phrases — use pos:"noun" instead).'
CONSISTENCY = '\nCONSISTENCY RULE — CRITICAL: The "correct" field is a 0-indexed integer pointing to the correct choice. The "explanation_ja" and "explanation_en" MUST refer to the SAME letter as "correct" (correct=0→A, correct=1→B, correct=2→C, correct=3→D). NEVER write "正解は(C)" if correct=3, NEVER write "The answer is B" if correct=0. ALWAYS verify the correct letter matches before responding.\nEXPLANATION FORMAT — MANDATORY: explanation_ja MUST start with "正解は(X)。" where X is the correct letter (A/B/C/D). Then explain why it is correct and why each wrong answer is wrong. Example: "正解は(B)。[理由]。(A)は[なぜ間違い]。(C)は[なぜ間違い]。(D)は[なぜ間違い]。"\nDISTRACTOR QUALITY — ALL PARTS: Wrong answers must NOT be valid answers to the question. Each wrong answer must fail for a clear, identifiable reason. If a wrong answer could reasonably be correct, replace it with something clearly wrong.'
AUDIO_RULE = '\nAUDIO CONSISTENCY RULE — CRITICAL: The "audio" field MUST contain the EXACT SAME text as what will be shown/read. For Part 1: audio MUST be verbatim "(A) <choice A text>. (B) <choice B text>. (C) <choice C text>. (D) <choice D text>." with the EXACT same wording as in the "choices" array. For Part 2: audio MUST be "<spoken question>. (A) <choice A>. (B) <choice B>. (C) <choice C>." with identical text. For Part 3: audio MUST be the EXACT "conversation" text. For Part 4: audio MUST be the EXACT "talk" text. NEVER use placeholders like [spoken] or [same as talk] — always write the literal text. If audio text differs from choices/conversation/talk, the output is INVALID.'

def get_level_rules(part, level):
    """Returns part-specific level rules (matches HTML level-specific guidance)."""
    rules = {
        "part1": {
            "beginner": '- Use ONLY simple present continuous active voice: "A man is reading a document."\n- Use basic verbs: sitting, standing, walking, reading, writing, carrying, looking at, holding\n- Distractors: wrong subject or completely wrong action (easy to eliminate)\n- Scene: 1 person doing 1 clear simple action',
            "intermediate": '- Mix active and passive voice: "A woman is examining a document." / "Some boxes are stacked near the wall."\n- Include preposition traps: "on the table" vs "under the table"\n- Include similar-sound traps: "filing" vs "filling", "waiting" vs "weighting"\n- Scene: 1-2 people with multiple objects in the background',
            "advanced": '- Use passive voice states and perfect passive: "The vehicle has been parked along the curb." / "Merchandise is being displayed."\n- Require distinguishing ongoing action vs completed state: "are being arranged" vs "have been arranged"\n- Distractors: each must be wrong for ONE clear reason — (1) wrong SUBJECT (describes different person/object), (2) wrong ACTION (person is doing something else), (3) wrong STATE (ongoing vs completed: "are being stacked" vs "have been stacked"), (4) wrong PREPOSITION/LOCATION ("on" vs "next to")\n- The difficulty is in the CORRECT answer using advanced grammar, NOT in making distractors also seem correct\n- Use advanced vocabulary: scaffolding, docked, paved, stacked, mounted, overlooking\n- Scene: complex scene with multiple people/objects where details matter',
        },
        "part2": {
            "beginner": '- Question: simple WH-question or basic Yes/No with common workplace vocabulary\n- Correct answer: DIRECT response. "Where is the copy room?" → "Down the hall on your left."\n- All responses are short spoken fragments (3-8 words)\n- Distractors: ONE uses a repeated keyword from the question but wrong topic, ONE is clearly unrelated\n- Scenario: simple office situations — meeting time, supply room location, number of copies, office directions\n- Avoid: negative questions, tag questions, embedded questions, statements, indirect responses',
            "intermediate": '- Question: WH, Yes/No, suggestions ("Why don\'t we...?"), offers, and choice ("A or B?")\n- Correct answer: sometimes INDIRECT (about 40% of the time)\n- Indirect patterns to use: (1) situation-hint — "Did you finish the report?" → "I\'ve been in meetings all day." (implies no), (2) referral — "Who is leading the project?" → "Check the internal memo." (implies answer is there), (3) counter-question — "Would you like something to drink?" → "Do you have orange juice?"\n- Distractors: use similar-sound traps ("filed"/"filled") and related-word traps ("meeting" in question → "conference" in wrong answer about different topic)\n- Scenario: standard business — project progress, client meeting, business trip, deadline, schedule change',
            "advanced": '- Question: negative questions ("Shouldn\'t we have...?"), tag questions ("The shipment arrived, didn\'t it?"), embedded questions ("Do you know when...?"), statements requiring response ("The printer seems jammed.")\n- Correct answer: predominantly INDIRECT (about 70%) — the difficulty is in UNDERSTANDING the implied meaning\n- Indirect response patterns (use variety):\n  (1) REFERRAL: "When is the training?" → "Susan is coordinating it." (= ask her)\n  (2) UNKNOWN/UNDECIDED: "When will the project start?" → "It hasn\'t been finalized yet."\n  (3) SITUATION HINT: "Are you going to the reception?" → "I have a deadline tonight." (= no)\n  (4) PREMISE DENIAL: "Did you submit the monthly report?" → "We only submit it quarterly." (= no monthly report exists)\n  (5) COUNTER-QUESTION: "Shouldn\'t we have received the shipment?" → "Who did you speak with at the warehouse?"\n  (6) THIRD OPTION: "Should we meet Monday or Tuesday?" → "Actually, let\'s do it by email."\n- Distractors: classic TOEIC traps — (1) WORD REPETITION: repeats keyword from question but wrong topic ("quarterly figures" → "quarterly meeting"), (2) TOPIC SHIFT: sounds work-related but addresses completely different subject, (3) WRONG CONTEXT: grammatically fine but logically unrelated\n- Scenario: standard TOEIC business — budget review, deadline confirmation, staff meeting, client follow-up, schedule coordination. NOT exotic vocabulary — difficulty comes from response indirectness, not word difficulty',
        },
        "part3": {
            "beginner": '- Questions: all 3 should be straightforward detail questions (Who/What/Where/When)\n- Answers are directly stated in the conversation — no inference needed\n- Wrong answers: mention things NOT said in the conversation, or attribute actions to the wrong speaker\n- Vocabulary in conversation: basic everyday business words',
            "intermediate": '- Questions: 2 detail + 1 inference ("What will the man probably do next?" or "What is the woman\'s concern?")\n- Some answers require paraphrasing — not using the exact words from the conversation\n- Wrong answers: use words from the conversation but about the wrong speaker or wrong detail\n- Vocabulary: standard business terms',
            "advanced": '- Questions: at least 2 inference/implied-meaning questions\n- MUST include one "What does the speaker mean when he/she says, \'...\'?" question\n- Answers heavily paraphrased or require synthesizing multiple statements\n- Correct answer: uses DIFFERENT WORDS from the conversation to express the same meaning (paraphrasing = difficulty)\n- Wrong answers: use EXACT WORDS from the conversation but answer a DIFFERENT question or describe the WRONG person\'s action\n- Example: conversation says "I\'ll handle the client meeting" → Correct: "He will attend to the appointment" (paraphrased) / Wrong: "He will handle the budget" (same verb, wrong object)\n- Vocabulary: sophisticated business language, idiomatic expressions',
        },
        "part4": {
            "beginner": '- Questions: 3 straightforward detail questions\n- Direct quotes from the talk in correct answers\n- Wrong answers: mention details NOT in the talk, or misquote numbers/dates\n- Vocabulary: basic monologue language',
            "intermediate": '- Questions: 2 detail + 1 inference\n- Some paraphrased answers\n- Wrong answers: use words from the talk but in the wrong context\n- Talk: includes signal words (however, additionally, in particular)',
            "advanced": '- Questions: at least 1 "What does the speaker imply when she says, \'...\'?" question\n- Talk uses formal language, subordinate clauses, implicit logical connections\n- Correct answer: paraphrased from the talk — requires understanding the meaning, not just matching words\n- Wrong answers: contain words/phrases from the talk but answer a different question or misrepresent the context\n- Include 1 detail question requiring careful listening (numbers, dates, conditions)',
        },
        "part5": {
            "beginner": '- Tests: simple word forms (noun/verb/adj/adv), basic prepositions, basic tenses\n- Distractors: clearly wrong word forms or tenses',
            "intermediate": '- Tests: vocabulary in context, conditionals, relative clauses, gerund/infinitive\n- Distractors: 2 partially-correct options that require careful reading',
            "advanced": '- Tests: subjunctive, inversion, parallel structure, advanced collocations\n- Grammar questions: only ONE answer fits the grammatical rule — distractors fail for identifiable grammar reasons (wrong tense, wrong form, wrong structure)\n- Vocabulary questions: 4 words with similar meanings, but only ONE fits the SPECIFIC context — wrong answers have wrong nuance or collocational mismatch\n- The difficulty is in KNOWING the grammar rule or precise word meaning, NOT in all choices being equally valid',
        },
        "part6": {
            "beginner": '- Blanks 1-3: simple word forms or vocabulary\n- Blank 4 (sentence): obvious choice based on context\n- Wrong answers: grammatically incorrect or clearly don\'t fit the sentence',
            "intermediate": '- Blanks 1-3: vocabulary in context, tense consistency\n- Blank 4: requires understanding the paragraph flow\n- Wrong answers: grammatically possible but semantically wrong for the context',
            "advanced": '- Blanks 1-3: subtle vocabulary distinctions, complex grammar\n- Blank 4: requires understanding the document\'s rhetorical structure\n- Wrong answers: near-synonyms that don\'t fit the specific context or tone',
        },
        "part7s": {
            "beginner": '- Questions: detail questions with direct quotes\n- Document: straightforward business email/notice with clear structure',
            "intermediate": '- Questions: mix of detail, purpose, and 1 inference\n- Document: business correspondence with moderate vocabulary',
            "advanced": '- Questions: include 1 NOT question and 1 inference question\n- Include vocabulary-in-context question\n- Document: formal report, legal notice, or multi-paragraph article',
        },
        "part7d": {
            "beginner": '- Questions: 4 detail + 1 simple cross-reference\n- No NOT questions; answers directly stated',
            "intermediate": '- Questions: 3 detail + 1 cross-reference + 1 inference\n- Documents: business correspondence pair with moderate vocabulary',
            "advanced": '- Questions: 2 cross-reference + 1 NOT + 1 inference + 1 detail\n- Documents: formal/complex pair with subtle connections',
        },
        "part7t": {
            "beginner": '- Questions: 3 detail + 2 simple cross-reference\n- Documents: clearly related with obvious connections',
            "intermediate": '- Questions: 2 detail + 2 cross-reference + 1 inference\n- Documents: business set with moderate vocabulary',
            "advanced": '- Questions: 2 cross-reference + 1 NOT + 1 vocabulary-in-context + 1 inference\n- Documents: complex set with multi-layered information',
        },
    }
    rule = rules.get(part, {}).get(level, "")
    return f"\nLEVEL-SPECIFIC RULES:\n{rule}\n" if rule else ""

def build_prompt(level, part, t):
    # Listening系(Part 1-4)では音声整合性ルールを追加
    is_listening = part in ("part1", "part2", "part3", "part3_3p", "part4")
    audio_rule = AUDIO_RULE if is_listening else ""
    phrase_rule = PHRASE_RULE if part not in ("part1", "part2") else ""
    sys = f"You are an expert TOEIC test maker. {LEVEL_GUIDES[level]}\nRespond with EXACTLY ONE JSON object — no arrays, no wrapping, no markdown, no backticks. DO NOT wrap the output in {{\"part1\":[...]}} or similar. DO NOT produce multiple questions. Output ONLY a single {{...}} object matching the template below.{JA}{EN_EXPL}{CONSISTENCY}{CHOICE_RULE}{audio_rule}{VOCAB_RULE}{phrase_rule}\nDIFFICULTY RATING: Include \"difficulty\" (integer 200-990). This question is being generated at {level.upper()} level.\nCALIBRATION: beginner→300-500, intermediate→450-700, advanced→650-950.\nCRITERIA: VOCAB(basic→400, business→650, advanced→850) + GRAMMAR(simple→400, clause→600, subjunctive→850) + INFERENCE(explicit→400, implied→650, cross-ref→800) + DISTRACTORS(obvious→400, plausible→650, tricky→850).\nCRITICAL: An advanced question MUST be rated 650+. Do NOT under-rate."
    tt, td = t.get("type","varied"), t.get("desc","")
    is_graphic = tt.startswith("graphic_")
    # Part 5 scenario diversity: randomly select a business context
    P5_SCENARIOS = [
        "a retail store placing a product order with a supplier",
        "an architect submitting a building renovation proposal",
        "a hotel manager coordinating a large conference booking",
        "a pharmaceutical company launching a new medication",
        "a city council approving a public park renovation",
        "a software startup pitching to venture capital investors",
        "a magazine editor reviewing article submissions",
        "a car dealership offering a seasonal promotion",
        "a hospital administrator updating patient intake procedures",
        "a museum curator organizing a special exhibition",
        "a shipping company optimizing delivery routes",
        "a restaurant chain expanding to a new city",
        "a university offering a new online degree program",
        "an insurance company processing a large claim",
        "a fashion brand preparing for a runway show",
        "a construction firm bidding on a highway project",
        "a veterinary clinic upgrading its appointment system",
        "a tech company releasing a security patch",
        "a law firm hiring summer associates",
        "a bakery catering a corporate event",
        "a real estate agent listing a commercial property",
        "a travel agency designing a group tour package",
        "a fitness center launching a membership drive",
        "a library system acquiring digital subscriptions",
        "an airline revising its frequent-flyer program",
        "a telecommunications company installing fiber optic lines",
        "a nonprofit organizing a charity auction",
        "a farming cooperative negotiating crop prices",
        "an accounting firm preparing quarterly tax filings",
        "a theater company selling season tickets",
        "a recycling plant upgrading sorting equipment",
        "a staffing agency placing temporary workers",
        "a dental office scheduling patient follow-ups",
        "a logistics company leasing warehouse space",
        "a radio station selling advertising slots",
        "a paint manufacturer testing new color formulas",
        "a government agency issuing environmental permits",
        "a sports arena hosting a music festival",
        "an engineering firm conducting a safety inspection",
        "a bookstore hosting an author signing event",
        "a florist preparing wedding arrangements",
        "a moving company quoting a residential relocation",
        "a photography studio booking portrait sessions",
        "a printing company fulfilling a large brochure order",
        "a consulting firm presenting audit findings to a client",
    ]
    p5_scenario = random.choice(P5_SCENARIOS)

    # Shared industry contexts for Part 3/6/7 diversity
    # Each entry: (industry, specific detail hint for realistic TOEIC content)
    INDUSTRY_CONTEXTS = [
        ("a publishing company", "book launches, manuscript deadlines, author signings, print runs"),
        ("an electronics manufacturer", "product testing, assembly lines, component suppliers, quality control"),
        ("a landscaping company", "garden designs, seasonal planting, client site visits, equipment maintenance"),
        ("an organic food company", "product sourcing, food safety certifications, store partnerships, packaging"),
        ("a dental clinic", "patient appointments, new equipment, hygiene protocols, insurance billing"),
        ("a furniture retailer", "showroom displays, delivery scheduling, warehouse inventory, custom orders"),
        ("an architectural firm", "building designs, city permits, client presentations, site inspections"),
        ("a pet supply store", "product ordering, grooming services, loyalty programs, seasonal promotions"),
        ("a catering company", "menu planning, event coordination, food preparation, vendor negotiations"),
        ("a solar energy company", "panel installations, government incentives, customer consultations, permits"),
        ("a language school", "class scheduling, teacher hiring, student enrollment, curriculum updates"),
        ("a car repair shop", "diagnostic services, parts ordering, customer estimates, warranty claims"),
        ("a textile factory", "fabric production, quality checks, export shipments, machinery upgrades"),
        ("a sports equipment brand", "product design, athlete endorsements, trade show exhibits, retail partners"),
        ("a wine distributor", "inventory management, restaurant clients, seasonal promotions, import regulations"),
        ("a coworking space", "membership plans, event hosting, facility upgrades, tenant requests"),
        ("a film production studio", "shooting schedules, location scouting, crew hiring, post-production"),
        ("a home cleaning service", "staff scheduling, supply ordering, client feedback, new service packages"),
        ("a biotech research lab", "experiment scheduling, grant applications, equipment procurement, safety reviews"),
        ("a jewelry store", "custom orders, gemstone sourcing, display arrangements, appraisal services"),
        ("an event planning agency", "venue bookings, vendor coordination, client consultations, budget tracking"),
        ("a freight shipping company", "route planning, customs documentation, fleet maintenance, client accounts"),
        ("a community center", "program scheduling, volunteer coordination, facility rentals, fundraising"),
        ("a mobile app developer", "sprint planning, bug tracking, user testing, app store submissions"),
        ("a plant nursery", "seasonal stock, greenhouse operations, wholesale orders, delivery logistics"),
        ("a private school", "enrollment, parent conferences, campus renovation, faculty meetings"),
        ("a luxury hotel chain", "guest services, staff training, renovation plans, loyalty rewards"),
        ("a package delivery service", "route optimization, driver scheduling, customer tracking, warehouse sorting"),
        ("a cosmetics brand", "product development, regulatory approval, retail launches, influencer partnerships"),
        ("an environmental consultancy", "site assessments, compliance reports, client advisories, field surveys"),
    ]
    _industry = random.choice(INDUSTRY_CONTEXTS)
    ctx_industry, ctx_details = _industry

    # Part 3: types that already imply a specific industry → skip injection
    P3_INDUSTRY_SPECIFIC = {
        "hotel_checkin","restaurant_order","retail_purchase","bank_finance",
        "medical_pharmacy","gym_membership","dry_cleaner","airport_travel",
        "car_rental","tour_booking","moving_relocation",
    }
    # Build industry instruction (only for generic Part 3 types)
    p3_ctx = ""
    if tt not in P3_INDUSTRY_SPECIFIC and not is_graphic:
        p3_ctx = f'\\nINDUSTRY CONTEXT: Set this conversation at {ctx_industry}. Use realistic details from this industry ({ctx_details}). Use specific names for people and places — avoid generic \"the company\" or \"the office\".'
    # Part 6/7 industry instruction (all types benefit)
    p67_ctx = f'\\nINDUSTRY CONTEXT: This document is from or about {ctx_industry}. Use realistic details ({ctx_details}). Include specific company/person names and industry-specific terms. Keep language natural and TOEIC-appropriate — no overly technical jargon.'
    # Graphic rule: appended to Part 3/4/7 prompts when type is graphic
    GRULE = ""
    if is_graphic:
        GRULE = '\\nGRAPHIC DATA — MANDATORY: You MUST include a "graphic" field in the JSON with structured data that the test-taker will see alongside the conversation/talk/text. The conversation/talk MUST reference specific data from this graphic. EXACTLY ONE of the 3 questions MUST start with "Look at the graphic." and can ONLY be answered by reading the graphic data.\\nGraphic format: "graphic":{{"title":"Graphic Title","headers":["Column1","Column2","Column3"],"rows":[["row1col1","row1col2","row1col3"],["row2col1","row2col2","row2col3"]]}}\\nMatch the graphic to the scenario type:\\n- Schedule/Agenda: headers=["Time","Event","Room"], rows=[["9:00 AM","Opening Remarks","Main Hall"],["10:00 AM","Workshop A","Room 201"]]\\n- Price list/Menu: headers=["Service","Standard","Premium"], rows=[["Oil Change","$35","$55"],["Tire Rotation","$25","$40"]]\\n- Order form/Invoice: headers=["Item","Qty","Unit Price","Total"], rows=[["Desk lamp","5","$35.00","$175.00"],["Monitor stand","3","$48.00","$144.00"]]\\n- Floor map/Seating: headers=["Room/Area","Department","Contact"], rows=[["Suite 301","Marketing","Ms. Chen"],["Suite 302","Finance","Mr. Park"]]\\n- Bar/Pie chart data: headers=["Quarter","Revenue ($M)","Growth (%)"], rows=[["Q1","12.4","8%"],["Q2","14.1","14%"],["Q3","11.8","-16%"]]\\n- Survey results: headers=["Category","Satisfaction (%)","Responses"], rows=[["Customer Service","92%","847"],["Delivery Speed","78%","823"]]\\n- Map/Directions: headers=["Destination","Building","Walking Time"], rows=[["Cafeteria","Building B","5 min"],["Parking","Lot C","8 min"]]\\n- Comparison: headers=["Feature","Plan A","Plan B","Plan C"], rows=[["Storage","10 GB","50 GB","Unlimited"],["Price/mo","$9.99","$19.99","$29.99"]]\\nUse 3-6 rows and 3-5 columns. Data must be SPECIFIC (real numbers, names, times) — never use placeholder text.'
    VEX = ',"difficulty":750,"vocab":[{{"word":"English word","pos":"noun","ja":"日本語","example":"Short sentence","level":"B1"}},{{"word":"in compliance with","pos":"phrase","ja":"〜に従って","example":"The project was completed in compliance with regulations.","level":"B2"}},{{"word":"word3","pos":"adjective","ja":"訳","example":"sentence","level":"C"}}]'
    # Get level-specific rules per part
    R1 = get_level_rules("part1", level)
    R2 = get_level_rules("part2", level)
    R3 = get_level_rules("part3", level)
    R4 = get_level_rules("part4", level)
    R5 = get_level_rules("part5", level)
    R6 = get_level_rules("part6", level)
    R7s = get_level_rules("part7s", level)
    R7d = get_level_rules("part7d", level)
    R7t = get_level_rules("part7t", level)
    B = {
        "part1": lambda: f'{sys}{R1}\nPart 1 (Photographs). SCENE: {tt} — {td}. 4 statements (A-D), 5-8 words each describing the photo objectively.\n- Correct answer: accurately describes what is visible.\n- Distractors: mention objects/actions that are NOT visible, wrong tense, or wrong subject.\nDO NOT include an "audio" field — it will be auto-generated from choices.\n{{"scene":"vivid 20-30 word description for image generation","choices":["(A) Five to eight words.","(B) Five to eight words.","(C) Five to eight words.","(D) Five to eight words."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答のトラップタイプ]","explanation_en":"Short 1-2 sentence English explanation"{VEX}}}',
        "part2": lambda: f'{sys}{R2}\nPart 2 (Question-Response). TYPE: {tt} — {td}. 3 responses (A-C). Correct answer is frequently INDIRECT (not a literal yes/no).\nVOCABULARY — CRITICAL FOR PART 2: Use NORMAL workplace English. Words like "meeting", "schedule", "report", "office", "delivery", "budget", "order" are correct. DO NOT use Part 5/7 vocabulary like "remuneration", "procurement", "notwithstanding", "forthcoming", "necessitate", "contingent upon", "arbitration". Part 2 difficulty comes from INDIRECT RESPONSES, not from exotic words.\nSCENARIO DIVERSITY — CRITICAL: Each question MUST be about a DIFFERENT workplace topic. Choose from: office supplies, meeting schedule, travel plans, lunch, parking, delivery, project deadline, new employee, equipment repair, training session, client visit, holiday schedule, building maintenance, job opening, company event. DO NOT repeat audit/compliance/regulatory themes.\nQUESTION TYPE COMPLIANCE — CRITICAL: The question MUST match the type "{tt}". If type is "yesno_do", use "Do/Does/Did". If "negative_isnt", use "Isn\'t/Aren\'t". If "wh_where_place", use "Where". Do NOT generate a different question type.\nDISTRACTOR VALIDITY — CRITICAL: Wrong answers must NOT be valid responses to the question. Test each: if someone said it in real conversation, would it make sense as a response? If yes, it is TOO GOOD for a wrong answer — change it. Wrong answers fail for: (1) answers a DIFFERENT question, (2) repeats a word but about a different topic, (3) completely unrelated subject. Example: Q="How long to deliver chairs?" GOOD wrong="The conference room is on the third floor." (unrelated) BAD wrong="The warehouse said next Friday." (this ANSWERS the question!)\nCRITICAL FORMAT: EXACTLY 3 choices (A)(B)(C). NEVER include (D). Each response MUST be 3-8 words (short spoken fragments).\nGood: "(A) In the conference room." / Bad: "(A) I believe the meeting was rescheduled to next Tuesday." (too long!)\nDO NOT include an "audio" field — it will be auto-generated.\n{{"spoken":"Natural question or statement 5-15 words","choices":["(A) 3-8 word response.","(B) 3-8 word response.","(C) 3-8 word response."],"correct":0,"explanation_ja":"【出題: {tt}】\\n和訳: (spoken の日本語訳)\\n正解理由と各誤答のトラップタイプを解説","explanation_en":"Short English"{VEX}}}',
        "part3": lambda: f'{sys}{R3}\nPart 3 (Conversations). SCENARIO: {tt} — {td}. "Man:"/"Woman:" labels. 5-8 turns, 60-100 words MAXIMUM. Keep conversation SHORT and natural. EXACTLY 3 questions. INCLUDE "translation_ja".{p3_ctx}{GRULE}\nGENDER RULES — STRICT:\n- "Man:" = MALE character (male names, he/him/his). "Woman:" = FEMALE character (female names, she/her).\n- In questions: "the man" = Man speaker, "the woman" = Woman speaker. NEVER swap.\n- translation_ja: Man = 男性, Woman = 女性. NEVER swap.\nDO NOT include an "audio" field — it will be auto-generated from conversation.\n{{"conversation":"Man: first line...\\nWoman: response...\\nMan: reply...\\nWoman: next...\\nMan: final...","translation_ja":"男性: ...\\n女性: ...","speakers":["Man","Woman"],"questions":[{{"question":"Where most likely are the speakers?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"What does the man/woman suggest?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"What will the speaker most likely do next?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":2,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}}]{VEX}}}',
        "part3_3p": lambda: f'{sys}{R3}\nPart 3 (Conversations) with EXACTLY 3 speakers.\nFORMAT: Choose "Man 1:", "Man 2:", "Woman:" OR "Woman 1:", "Woman 2:", "Man:"\nSCENARIO: {tt} — {td}. 7-10 turns, 80-120 words. All 3 speakers must have at least 2 turns. EXACTLY 3 questions.{p3_ctx}\n\n3-PERSON CONVERSATION PATTERNS — TOEIC公式準拠. Use ONE of these patterns:\n1. BRIEF THIRD SPEAKER (電話交換手/受付型): A&B are talking → third person briefly appears as operator, receptionist, or assistant with 1-2 short lines. e.g. "One moment, I\'ll transfer you." → "Hello, shipping department, how can I help?"\n2. SAME-ROLE PAIR (同立場ペア型): Two same-gender speakers share the same position/stance and interact with the different-gender speaker. e.g. Two coworkers invite a colleague to lunch, or two team members report to a manager. The pair may say "We both think..." or take turns explaining.\n3. INTRODUCTION/JOINING (紹介・合流型): A&B discuss a topic → one introduces the third. e.g. "Oh, here comes [name] from IT." or "Let me introduce [name] — she handles our accounts." Third person joins with relevant information.\n4. CUSTOMER + TWO STAFF (顧客＋2スタッフ型): Customer talks to Staff A → Staff A cannot fully help → refers to Staff B. e.g. "My colleague handles warranty claims. [Name], could you help?" → Staff B takes over.\n5. SEQUENTIAL CONSULTATION (相談リレー型): A has a problem → asks B → B says "Let\'s check with [name]" → C provides the answer/solution.\n6. THREE-WAY MEETING (3者ミーティング型): Three colleagues in a meeting, each with a distinct role (marketing/finance/operations). Each contributes their department\'s perspective.\n\nDESIGN PRINCIPLE: The third speaker must be introduced naturally so listeners can follow who is speaking. Two same-gender speakers should have clearly distinct roles or viewpoints.\n\nGENDER RULES — STRICT:\n- "Man 1:"/"Man 2:" = MALE. "Woman:"/"Woman 1:"/"Woman 2:" = FEMALE. NEVER swap.\n- translation_ja: Man 1 = 男性1, Man 2 = 男性2, Woman = 女性. NEVER swap.\nDO NOT include an "audio" field — it will be auto-generated.\n{{"conversation":"Man 1: ...\\nWoman: ...\\nMan 2: ...","translation_ja":"男性1: ...\\n女性: ...\\n男性2: ...","speakers":["Man 1","Man 2","Woman"],"questions":[{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":2,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}}]{VEX}}}',
        "part4": lambda: f'{sys}{R4}\nPart 4 (Talks). TYPE: {tt} — {td}. Single-speaker monologue, 100-140 words, 6-10 sentences. EXACTLY 3 questions. INCLUDE "translation_ja".{GRULE}\nDO NOT include an "audio" field — it will be auto-generated from talk.\n{{"talk":"Full monologue 100-140 words...","translation_ja":"トーク全文の日本語訳","talk_type":"{tt}","questions":[{{"question":"What is the purpose of the message/announcement?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"What does the speaker imply?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"What are listeners asked to do?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":2,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}}]{VEX}}}',
        "part5": lambda: f'{sys}{BLANK_RULE}{R5}\nPart 5 (Incomplete Sentences). CATEGORY: {tt} — {td}.\n\nSCENARIO CONTEXT: Write the sentence about {p5_scenario}. DO NOT use generic "the company" or "the department" — use specific names, roles, or details from this scenario.\n\nRULES:\n1. Write a business sentence (15-25 words) with EXACTLY ONE blank: -------\n2. The ------- replaces the tested word. Without it, the output is INVALID.\n3. All 4 choices (A-D) must be plausible.\n4. "correct" = index of the right answer (0=A, 1=B, 2=C, 3=D).\n5. explanation_ja MUST name the correct letter first: "正解は(X)..." where X matches "correct".\n6. INCLUDE "translation_ja" — the full sentence with the correct answer filled in, translated to Japanese.\n7. explanation_ja MUST explain WHY EACH WRONG CHOICE is incorrect.\n\nGOOD: "The ------- of the new policy was announced yesterday."\nBAD: "The implementation of the new policy was announced yesterday." (NO BLANK = REJECTED)\n\n{{"sentence":"The manager asked all employees to ------- the updated safety guidelines before Friday.","choices":["(A) review","(B) reviewing","(C) reviewed","(D) reviewer"],"correct":0,"translation_ja":"マネージャーは全従業員に、金曜日までに更新された安全ガイドラインを確認するよう求めた。","explanation_ja":"正解は(A) review。ask + 人 + to + 動詞原形の形。(B) reviewingはing形で不可、(C) reviewedは過去形で不可、(D) reviewerは名詞「評論家」で文意に合わない。","explanation_en":"ask someone to + base verb"{VEX}}}',
        "part6": lambda: f'{sys}{BLANK_RULE6}{R6}\nPart 6 (Text Completion). DOC TYPE: {tt} — {td}.{p67_ctx}\n150-200 words with EXACTLY 4 blanks: (1)------- (2)------- (3)------- (4)-------.\nBlanks 1-3: word/phrase choices. Blank 4: SENTENCE INSERTION (choices are full sentences).\nINCLUDE "translation_ja" (with answers filled in).\n{{"doc_type":"{tt}","header":"To: ...\\nFrom: ...\\nSubject: ...","text":"Full text 150-200 words with (1)------- and (2)------- and (3)------- and (4)-------","translation_ja":"日本語訳（空所に正解が入った状態）","questions":[{{"blank":1,"question":"Context around blank (1)","choices":["(A) word","(B) word","(C) word","(D) word"],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"blank":2,"question":"Context around blank (2)","choices":["(A) word","(B) word","(C) word","(D) word"],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"blank":3,"question":"Context around blank (3)","choices":["(A) word","(B) word","(C) word","(D) word"],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"blank":4,"question":"Which sentence best fits?","choices":["(A) Full sentence A.","(B) Full sentence B.","(C) Full sentence C.","(D) Full sentence D."],"correct":2,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}}]{VEX}}}',
        "part7s": lambda: f'{sys}{R7s}\nPart 7 Single Passage. DOC TYPE: {tt} — {td}. 150-250 words. Generate 2-4 questions.{p67_ctx}{GRULE}\nINCLUDE "translation_ja".\n{{"doc_type":"{tt}","header":"document header if applicable","text":"150-250 word passage","translation_ja":"文書全文の日本語訳","questions":[{{"question":"What is the main purpose?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"What is indicated/suggested about X?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"According to the document, what...?","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":2,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}}]{VEX}}}',
        "part7d": lambda: f'{sys}{R7d}\nPart 7 Double Passage. PAIR: {tt} — {td}. Two related documents, 100-180 words each. EXACTLY 5 questions, including at least 1 CROSS-REFERENCE question requiring info from BOTH documents.{p67_ctx}\nINCLUDE "translation_ja_1","translation_ja_2".\n{{"doc_type_1":"email/notice/memo","header_1":"...","text_1":"100-180 words","translation_ja_1":"日本語訳1","doc_type_2":"reply/schedule","header_2":"...","text_2":"100-180 words","translation_ja_2":"日本語訳2","questions":[{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"Cross-reference: based on BOTH documents, ...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":2,"explanation_ja":"正解は(X)。[クロスリファレンス解説]","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}}]{VEX}}}',
        "part7t": lambda: f'{sys}{R7t}\nPart 7 Triple Passage. SET: {tt} — {td}. Three related documents, 80-150 words each. EXACTLY 5 questions, including at least 2 CROSS-REFERENCE questions requiring info from multiple documents.{p67_ctx}\nINCLUDE "translation_ja_1","translation_ja_2","translation_ja_3".\n{{"doc_type_1":"...","header_1":"...","text_1":"80-150 words","translation_ja_1":"日本語訳1","doc_type_2":"...","header_2":"...","text_2":"80-150 words","translation_ja_2":"日本語訳2","doc_type_3":"...","header_3":"...","text_3":"80-150 words","translation_ja_3":"日本語訳3","questions":[{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"Cross-reference: based on docs 1+2 (or 1+3), ...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":2,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]（クロスリファレンス）","explanation_en":"Short English"}},{{"question":"...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":0,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]","explanation_en":"Short English"}},{{"question":"Cross-reference: based on docs 2+3 (or 1+2+3), ...","choices":["(A) ...","(B) ...","(C) ...","(D) ..."],"correct":1,"explanation_ja":"正解は(X)。[正解の理由]。[各誤答が間違いの理由]（クロスリファレンス）","explanation_en":"Short English"}}]{VEX}}}',
    }
    if part == "part7":
        sub = random.choice(["part7s","part7d","part7t"])
        return B[sub](), sub
    if part == "part3":
        # 本番TOEIC: 13会話中3つが3人会話（約23%）
        # ただし「3人+図表」の組み合わせは出題されない
        if not is_graphic:
            sub = "part3_3p" if random.random() < 0.25 else "part3"
        else:
            sub = "part3"
        return B[sub](), "part3"
    return B.get(part, B["part5"])(), part

# ══════════════════════════════════════
# JSON Parse + Normalize
# ══════════════════════════════════════
def parse_json(text):
    t = re.sub(r'```json|```', '', text).strip()
    def _fix_str(s):
        out, in_str, esc = [], False, False
        for c in s:
            if esc: out.append(c); esc = False; continue
            if c == '\\': esc = True; out.append(c); continue
            if c == '"': in_str = not in_str; out.append(c); continue
            if in_str:
                if c == '\n': out.append('\\n'); continue
                if c == '\r': continue
                if c == '\t': out.append('\\t'); continue
            out.append(c)
        return ''.join(out)
    def _unwrap(result):
        """Handle various nesting patterns LLMs produce:
        - [obj, obj] → obj (first)
        - {"part1": [obj, obj]} → obj (first of first key)
        - {"part1": obj, "part2": obj} → obj (first value)
        - {"questions": [...], ...} → keep as-is (normal)
        """
        if isinstance(result, list):
            if not result: raise ValueError("Empty array")
            print(f"[PARSE] Array ({len(result)} items), using first", flush=True)
            return result[0]
        if isinstance(result, dict):
            # If dict has only partN keys (nested structure), unwrap
            keys = list(result.keys())
            part_keys = [k for k in keys if re.match(r'^part[1-7][a-z_]*$', k)]
            # Case: {"part1": [obj, obj]} or {"part1": obj}
            if part_keys and all(k in part_keys for k in keys):
                first_key = part_keys[0]
                inner = result[first_key]
                print(f"[PARSE] Nested structure {{{first_key}: ...}}, unwrapping", flush=True)
                if isinstance(inner, list):
                    if not inner: raise ValueError("Empty nested array")
                    return inner[0]
                return inner
        return result
    # Try direct parse
    for attempt_text in [t, _fix_str(t)]:
        try:
            result = json.loads(attempt_text)
            return _unwrap(result)
        except json.JSONDecodeError:
            continue
    # Try to extract JSON object from text
    match = re.search(r'\{', t)
    if match:
        depth = 0
        start = match.start()
        for i in range(start, len(t)):
            if t[i] == '{': depth += 1
            elif t[i] == '}': depth -= 1
            if depth == 0:
                try:
                    result = json.loads(_fix_str(t[start:i+1]))
                    print(f"[PARSE] Extracted JSON object from text", flush=True)
                    return _unwrap(result)
                except: break
    # Last resort: try truncating at last complete property
    fixed = _fix_str(t)
    for end_pattern in ['"}]}', '"}]', '"}', '"}}}']:
        idx = fixed.rfind(end_pattern)
        if idx > 0:
            candidate = fixed[:idx+len(end_pattern)]
            # Close any open brackets
            opens = candidate.count('{') - candidate.count('}')
            closes = candidate.count('[') - candidate.count(']')
            candidate += ']' * max(closes, 0) + '}' * max(opens, 0)
            try:
                result = json.loads(candidate)
                print(f"[PARSE] Repaired truncated JSON", flush=True)
                return _unwrap(result)
            except: continue
    raise ValueError(f"JSON parse failed at char ~{len(t)}")

def strip_label(c, labels="A-D"):
    return re.sub(r'^\([' + labels + r']\)\s*', '', c)

def check_answer_consistency(qs, part):
    """Detect if explanation_ja/_en mentions a different answer letter than 'correct'.
    Attempts auto-fix: if explanation clearly states a different letter, update 'correct' to match."""
    questions = qs.get("questions", [])
    import re
    all_ok = True
    for qi, q in enumerate(questions):
        correct = q.get("correct", 0)
        if not isinstance(correct, int) or correct < 0 or correct > 3:
            continue
        correct_letter = ["A","B","C","D"][correct]
        expl_ja = q.get("explanation_ja","").strip()
        expl_en = q.get("explanation_en","").strip()
        # Check Japanese explanation — broad patterns
        # Match: 正解は(A), 正解は（B）, 正解：(C), 正解理由：Aは, 答え：B
        ja_patterns = [
            r'(?:正解は|答えは|正解[:：]\s*|正解理由[:：]\s*)\s*[\(（]?([A-D])[\)）]?',
            r'[\(（]([A-D])[\)）]\s*(?:が正解|が正しい|が適切)',
        ]
        mentioned_ja = None
        for pat in ja_patterns:
            m = re.search(pat, expl_ja)
            if m:
                mentioned_ja = m.group(1).upper()
                break
        ja_confirmed = False
        if mentioned_ja:
            if mentioned_ja != correct_letter:
                new_correct = {"A":0,"B":1,"C":2,"D":3}.get(mentioned_ja)
                if new_correct is not None:
                    print(f"[FIX] Q{qi+1} {part}: correct={correct_letter}→{mentioned_ja} (matched explanation_ja)", flush=True)
                    q["correct"] = new_correct
                    ja_confirmed = True
                else:
                    all_ok = False
            else:
                ja_confirmed = True  # Japanese matches correct — trusted
        # Check English explanation — only if Japanese didn't confirm
        if not ja_confirmed:
            en_patterns = [
                r'(?:answer is|correct (?:answer )?is)\s*\(?([A-D])\)?',
                r'\(?([A-D])\)?\s*is (?:the )?correct',
            ]
            mentioned_en = None
            for pat in en_patterns:
                m = re.search(pat, expl_en, re.I)
                if m:
                    mentioned_en = m.group(1).upper()
                    break
            if mentioned_en:
                new_letter = ["A","B","C","D"][q.get("correct",0)]
                if mentioned_en != new_letter:
                    new_correct = {"A":0,"B":1,"C":2,"D":3}.get(mentioned_en)
                    if new_correct is not None:
                        print(f"[FIX] Q{qi+1} {part}: correct→{mentioned_en} (matched explanation_en)", flush=True)
                        q["correct"] = new_correct
                    else:
                        all_ok = False
    return all_ok

def normalize_set(raw, part):
    vocab = raw.get("vocab", [])  # Preserve vocabulary from LLM output
    if part == "part1":
        ch = raw.get("choices", [])
        # ALWAYS rebuild audio from choices to avoid LLM mismatches
        # Strip "(A) " prefix from choices and reconstruct
        clean_choices = [strip_label(c) for c in ch]
        audio = " ... ".join(f"({chr(65+i)}) {clean_choices[i].rstrip('.')}." for i in range(len(clean_choices)))
        # Detect placeholder patterns in LLM's audio output (sanity check, not used but logged)
        llm_audio = raw.get("audio", "")
        if llm_audio and "[" in llm_audio and "]" in llm_audio:
            print(f"[WARN] Part1: LLM audio contained placeholders, using reconstructed audio", flush=True)
        return {"part":part,"scene":raw.get("scene",""),"audio":audio,"vocab":vocab,
                "questions":[{"question":"Which statement best describes the photograph?",
                              "choices":ch,"correct":raw.get("correct",0),
                              "explanation_ja":raw.get("explanation_ja",""),
                              "explanation_en":raw.get("explanation_en","")}]}
    if part == "part2":
        ch, sp = raw.get("choices",[]), raw.get("spoken","")
        # ALWAYS rebuild audio from spoken + choices
        clean_choices = [strip_label(c, "A-C") for c in ch]
        audio = sp + " ... " + " ... ".join(f"({chr(65+i)}) {clean_choices[i].rstrip('.')}." for i in range(len(clean_choices)))
        llm_audio = raw.get("audio", "")
        if llm_audio and "[" in llm_audio and "]" in llm_audio:
            print(f"[WARN] Part2: LLM audio contained placeholders, using reconstructed audio", flush=True)
        return {"part":part,"spoken":sp,"audio":audio,"vocab":vocab,
                "questions":[{"question":sp,"choices":ch,"correct":raw.get("correct",0),
                              "explanation_ja":raw.get("explanation_ja",""),
                              "explanation_en":raw.get("explanation_en","")}]}
    if part in ("part3", "part3_3p"):
        # ALWAYS use conversation as audio (avoid LLM placeholder issues)
        conv = raw.get("conversation","")
        llm_audio = raw.get("audio", "")
        if llm_audio and llm_audio.strip() != conv.strip():
            if "[" in llm_audio or len(llm_audio) < len(conv) * 0.5:
                print(f"[WARN] Part3: LLM audio differs from conversation, using conversation", flush=True)
        return {"part":"part3","conversation":conv,"translation_ja":raw.get("translation_ja"),
                "audio":conv,"speakers":raw.get("speakers",["Man","Woman"]),
                "graphic":raw.get("graphic"),"vocab":vocab,"questions":raw.get("questions",[])}
    if part == "part4":
        # ALWAYS use talk as audio
        talk = raw.get("talk","")
        llm_audio = raw.get("audio", "")
        if llm_audio and llm_audio.strip() != talk.strip():
            if "[" in llm_audio or len(llm_audio) < len(talk) * 0.5:
                print(f"[WARN] Part4: LLM audio differs from talk, using talk", flush=True)
        return {"part":part,"talk":talk,"translation_ja":raw.get("translation_ja"),
                "talk_type":raw.get("talk_type",""),"audio":talk,
                "graphic":raw.get("graphic"),"vocab":vocab,"questions":raw.get("questions",[])}
    if part == "part5":
        s = raw.get("sentence","")
        if "-------" not in s:
            # Try to fix: insert blank at correct answer position
            correct = raw.get("correct", 0)
            choices = raw.get("choices", [])
            correct_word = re.sub(r'^\([A-D]\)\s*', '', choices[correct] if correct < len(choices) else "")
            if correct_word and correct_word in s:
                s = s.replace(correct_word, "-------", 1)
                print(f"[Part5] Fixed missing blank: inserted ------- for '{correct_word}'", flush=True)
            else:
                print(f"[WARN] Part5: no blank in sentence and cannot auto-fix!", flush=True)
                raise ValueError("Part 5: sentence has no blank (-------)")
        return {"part":part,"vocab":vocab,"translation_ja":raw.get("translation_ja",""),"questions":[{"question":s,"choices":raw.get("choices",[]),"correct":raw.get("correct",0),"explanation_ja":raw.get("explanation_ja",""),"explanation_en":raw.get("explanation_en","")}]}
    if part == "part6":
        t = raw.get("text","")
        if "-------" not in t: print(f"[WARN] Part6: no blanks in text!", flush=True)
        return {"part":part,"doc_type":raw.get("doc_type",""),"header":raw.get("header"),"text":t,"translation_ja":raw.get("translation_ja"),"vocab":vocab,"questions":raw.get("questions",[])}
    if raw.get("text_1") and raw.get("text_2") and raw.get("text_3"):
        return {"part":"part7","isTriple":True,"doc_type_1":raw.get("doc_type_1",""),"header_1":raw.get("header_1"),"text_1":raw.get("text_1",""),"translation_ja_1":raw.get("translation_ja_1"),"doc_type_2":raw.get("doc_type_2",""),"header_2":raw.get("header_2"),"text_2":raw.get("text_2",""),"translation_ja_2":raw.get("translation_ja_2"),"doc_type_3":raw.get("doc_type_3",""),"header_3":raw.get("header_3"),"text_3":raw.get("text_3",""),"translation_ja_3":raw.get("translation_ja_3"),"vocab":vocab,"questions":raw.get("questions",[])}
    if raw.get("text_1") and raw.get("text_2"):
        return {"part":"part7","isDouble":True,"doc_type_1":raw.get("doc_type_1",""),"header_1":raw.get("header_1"),"text_1":raw.get("text_1",""),"translation_ja_1":raw.get("translation_ja_1"),"doc_type_2":raw.get("doc_type_2",""),"header_2":raw.get("header_2"),"text_2":raw.get("text_2",""),"translation_ja_2":raw.get("translation_ja_2"),"vocab":vocab,"questions":raw.get("questions",[])}
    return {"part":"part7","isDouble":False,"doc_type":raw.get("doc_type",""),"header":raw.get("header"),"text":raw.get("text",""),"translation_ja":raw.get("translation_ja"),"graphic":raw.get("graphic"),"vocab":vocab,"questions":raw.get("questions",[])}

def enforce_choice_count(qs):
    """Enforce TOEIC choice count: Part 2 = 3 choices (A-C), all others = 4 choices (A-D).
    Truncates extra choices and adjusts 'correct' index if needed."""
    part = qs.get("part", "")
    max_choices = 3 if part == "part2" else 4
    for q in qs.get("questions", []):
        choices = q.get("choices", [])
        if len(choices) > max_choices:
            print(f"[FIX] {part}: truncated {len(choices)} choices → {max_choices}", flush=True)
            q["choices"] = choices[:max_choices]
            # Ensure correct index is within range
            if q.get("correct", 0) >= max_choices:
                q["correct"] = 0
                print(f"[FIX] {part}: correct index out of range, reset to 0", flush=True)
    return qs


def shuffle_answer_positions(qs):
    """Randomly shuffle choice order so correct answer isn't always (A).
    Updates 'correct' index and ALL label references in explanations."""
    labels = ["(A)","(B)","(C)","(D)"]
    fw_labels = ["（A）","（B）","（C）","（D）"]
    for q in qs.get("questions", []):
        choices = q.get("choices", [])
        correct_idx = q.get("correct", 0)
        if not choices or correct_idx >= len(choices):
            continue
        n = len(choices)
        order = list(range(n))
        random.shuffle(order)
        new_choices = [None] * n
        new_correct = 0
        old_to_new = {}  # old_pos → new_pos
        for new_pos, old_pos in enumerate(order):
            c = re.sub(r'^\([A-D]\)\s*', '', choices[old_pos])
            new_choices[new_pos] = f"{labels[new_pos]} {c}"
            if old_pos == correct_idx:
                new_correct = new_pos
            old_to_new[old_pos] = new_pos
        q["choices"] = new_choices
        q["correct"] = new_correct
        # Remap ALL label references using placeholders to avoid collision
        needs_remap = any(old != new for old, new in old_to_new.items())
        if needs_remap:
            for field in ["explanation_ja", "explanation_en"]:
                if field not in q or not q[field]:
                    continue
                text = q[field]
                # Step 1: All labels → placeholders
                for i in range(n):
                    text = text.replace(labels[i], f"__LABEL{i}__")
                    text = text.replace(fw_labels[i], f"__FWLABEL{i}__")
                # Step 2: Placeholders → remapped labels
                for old_pos, new_pos in old_to_new.items():
                    text = text.replace(f"__LABEL{old_pos}__", labels[new_pos])
                    text = text.replace(f"__FWLABEL{old_pos}__", fw_labels[new_pos])
                q[field] = text
    # Rebuild audio field to match shuffled choices (Part 1/2 include choices in audio)
    part = qs.get("part","")
    if part == "part1" and qs.get("questions"):
        ch = qs["questions"][0].get("choices",[])
        cleaned = [strip_label(c).rstrip('.') for c in ch]
        qs["audio"] = " ... ".join(f"({chr(65+i)}) {cleaned[i]}." for i in range(len(cleaned)))
    elif part == "part2" and qs.get("spoken") and qs.get("questions"):
        sp = qs["spoken"]
        ch = qs["questions"][0].get("choices",[])
        cleaned = [strip_label(c, "A-C").rstrip('.') for c in ch]
        qs["audio"] = sp + " ... " + " ... ".join(f"({chr(65+i)}) {cleaned[i]}." for i in range(len(cleaned)))
    return qs

# ══════════════════════════════════════
# Text Generation
# ══════════════════════════════════════
def generate_text(prompt, engine, model, ollama_url, api_key):
    t0 = time.time()
    if engine == "ollama":
        print(f"[OLLAMA] {model} prompt={len(prompt)}c waiting...", flush=True)
        timeout = 600 if "gemma4" in model else 300  # gemma4 thinking needs more time
        resp = requests.post(f"{ollama_url}/api/generate", json={
            "model":model,"prompt":prompt,"stream":False,
            "options":{"temperature":0.7,"num_predict":4096,"num_gpu":99}
        }, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[OLLAMA] {resp.status_code} ({elapsed:.1f}s)", flush=True)
        resp.raise_for_status()
        result = resp.json()["response"]
        # Strip thinking content if present
        result = re.sub(r'<\|?think(?:ing)?\|?>.*?<\|?/think(?:ing)?\|?>', '', result, flags=re.DOTALL).strip()
    else:
        if not api_key: raise RuntimeError("Gemini API key required")
        print(f"[GEMINI] {model} prompt={len(prompt)}c", flush=True)
        body = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.7,"maxOutputTokens":4096,"thinkingConfig":{"thinkingBudget":GEMINI_THINKING.get(model,0)}}}
        # Retry on 429/500/503 with exponential backoff
        max_retries = 4
        for attempt in range(max_retries):
            resp = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",json=body,timeout=120)
            elapsed = time.time() - t0
            print(f"[GEMINI] {resp.status_code} ({elapsed:.1f}s)", flush=True)
            if resp.status_code in (429, 500, 503) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                print(f"[GEMINI] {resp.status_code}, retry in {wait}s ({attempt+1}/{max_retries-1})", flush=True)
                time.sleep(wait)
                t0 = time.time()
                continue
            break
        resp.raise_for_status()
        parts = resp.json().get("candidates",[{}])[0].get("content",{}).get("parts",[])
        result = "".join(p.get("text","") for p in parts if "text" in p)
    print(f"[GEN] {len(result)}c: {result[:150]}", flush=True)
    if len(result) > 10000:
        print(f"[WARN] Response is unusually long ({len(result)}c) - LLM may have generated multiple questions", flush=True)
    return result

def ollama_warmup(url, model):
    print(f"[WARMUP] {model}...", flush=True)
    try:
        t0 = time.time()
        requests.post(f"{url}/api/generate",json={"model":model,"prompt":"Hi","stream":False,"options":{"num_predict":1,"num_gpu":99}},timeout=300)
        print(f"[WARMUP] OK ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[WARMUP] Failed: {e}", flush=True)

# ══════════════════════════════════════
# ══════════════════════════════════════
# Text preprocessing for TTS (used by Edge/Gemini)
# ══════════════════════════════════════
def is_female(label):
    return bool(re.search(r'\bwoman\b|\bfemale\b|\bms\b|\bmrs\b', label, re.I)) or label.lower().startswith("woman")

def preprocess_tts_text(text):
    """Clean text for better TTS quality (works for Edge, Gemini, etc.):
    - Expand abbreviations (Mr. → Mister)
    - Normalize whitespace
    - Ensure sentence-ending punctuation
    """
    if not text: return ""
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Convert common abbreviations to spoken form
    abbrev = {
        r'\bMr\.\s': 'Mister ', r'\bMrs\.\s': 'Misses ',
        r'\bMs\.\s': 'Miss ', r'\bDr\.\s': 'Doctor ',
        r'\bSt\.\s': 'Saint ', r'\bvs\.\s': 'versus ',
        r'\be\.g\.\s': 'for example, ', r'\bi\.e\.\s': 'that is, ',
        r'\betc\.\s*': 'et cetera. ',
    }
    for pat, rep in abbrev.items():
        text = re.sub(pat, rep, text)
    # Ensure sentence-ending punctuation
    if text and text[-1] not in '.!?,;:"\')':
        text += '.'
    # Add a small space after sentence-ending punctuation if missing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    return text

def wav_to_opus(wav):
    try:
        r = subprocess.run(['ffmpeg','-i','pipe:','-c:a','libopus','-b:a','16k','-f','webm','pipe:'],input=wav,capture_output=True,timeout=30)
        return r.stdout if r.returncode==0 else None
    except FileNotFoundError: return None

# ── Azure Speech TTS (Microsoft Neural Voices, Paid) ──
AZURE_VOICES_F = ["en-US-AvaMultilingualNeural", "en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"]
AZURE_VOICES_M = ["en-US-AndrewMultilingualNeural", "en-US-GuyNeural", "en-US-DavisNeural", "en-GB-RyanNeural"]

_azure_token_cache = {"token": None, "expires": 0}

def _azure_get_token(key, region):
    """Get Azure access token (cached for 9 minutes)."""
    import time
    now = time.time()
    if _azure_token_cache["token"] and now < _azure_token_cache["expires"]:
        return _azure_token_cache["token"]
    ep = st.session_state.get("azure_speech_endpoint","").strip().rstrip("/")
    if ep:
        token_url = f"{ep}/sts/v1.0/issueToken"
    else:
        token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    resp = requests.post(token_url,
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Length": "0"},
        timeout=10)
    resp.raise_for_status()
    token = resp.text
    _azure_token_cache["token"] = token
    _azure_token_cache["expires"] = now + 540
    return token

def _azure_tts_endpoint():
    """Get Azure TTS synthesis endpoint."""
    ep = st.session_state.get("azure_speech_endpoint","").strip().rstrip("/")
    if ep:
        return f"{ep}/tts/cognitiveservices/v1"
    region = st.session_state.get("azure_speech_region","eastus")
    return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

def azure_tts(text, key, region, voice=None, rate="0%"):
    """Azure Speech TTS. Returns MP3 bytes."""
    if not voice:
        voice = random.choice(AZURE_VOICES_F + AZURE_VOICES_M)
    token = _azure_get_token(key, region)
    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
<voice name='{voice}'><prosody rate='{rate}'>{text}</prosody></voice></speak>"""
    resp = requests.post(
        _azure_tts_endpoint(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        },
        data=ssml.encode("utf-8"),
        timeout=30)
    resp.raise_for_status()
    return resp.content  # MP3 bytes

def azure_tts_conv(text, key, region, speakers=None):
    """Azure Speech multi-speaker TTS. Assigns fixed voice per speaker."""
    # Build speaker → voice mapping (consistent throughout conversation)
    spk = speakers or ["Man", "Woman"]
    fp, mp = AZURE_VOICES_F.copy(), AZURE_VOICES_M.copy()
    random.shuffle(fp); random.shuffle(mp)
    fi = mi = 0; voice_map = {}
    for s in spk:
        if s.lower().startswith("woman"):
            voice_map[s] = fp[fi % len(fp)]; fi += 1
        else:
            voice_map[s] = mp[mi % len(mp)]; mi += 1
    print(f"[AZURE-TTS] Voice map: {voice_map}", flush=True)

    lines = [l.strip() for l in text.replace("\\n","\n").split("\n") if l.strip()]
    all_mp3 = b""
    for line in lines:
        m = re.match(r'((?:Man|Woman|Speaker)\s*\d?)\s*:\s*(.*)', line, re.I)
        if m:
            label = m.group(1).strip()
            spoken = m.group(2).strip()
            # Find voice from map (try exact match, then prefix match)
            voice = voice_map.get(label)
            if not voice:
                for k, v in voice_map.items():
                    if k.lower().startswith(label.lower()[:3]):
                        voice = v; break
            if not voice:
                voice = random.choice(AZURE_VOICES_F + AZURE_VOICES_M)
        else:
            voice = random.choice(AZURE_VOICES_F + AZURE_VOICES_M)
            spoken = line
        if spoken:
            mp3 = azure_tts(spoken, key, region, voice)
            all_mp3 += mp3
    return all_mp3 if all_mp3 else None

# Gemini TTS
# ── Edge TTS (Microsoft Neural Voices, Free) ──
_edge_ok = None
def check_edge_tts():
    global _edge_ok
    if _edge_ok is not None: return _edge_ok
    try:
        import edge_tts
        _edge_ok = True
    except ImportError:
        _edge_ok = False
    return _edge_ok

# 高品質な Edge TTS 英語音声（学習用に厳選）
EDGE_VF = [
    "en-US-AriaNeural",        # 女性・ニュース系・最高品質
    "en-US-JennyNeural",       # 女性・カジュアル
    "en-US-AvaMultilingualNeural",  # 女性・最新2024
    "en-GB-SoniaNeural",       # イギリス英語女性
]
EDGE_VM = [
    "en-US-GuyNeural",         # 男性・ビジネス
    "en-US-DavisNeural",       # 男性・カジュアル
    "en-US-AndrewMultilingualNeural",  # 男性・最新2024
    "en-GB-RyanNeural",        # イギリス英語男性
]

def edge_tts_sync(text, voice="en-US-AriaNeural", rate="+0%"):
    """Synchronous wrapper for edge-tts. Returns MP3 bytes.
    Always runs in a fresh event loop in a dedicated thread to avoid
    Streamlit's event loop conflicts."""
    import asyncio, edge_tts, concurrent.futures
    cleaned = preprocess_tts_text(text)
    async def _gen():
        comm = edge_tts.Communicate(cleaned, voice, rate=rate)
        chunks = []
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        if not chunks:
            raise RuntimeError("Edge TTS returned no audio chunks")
        return b"".join(chunks)
    def _run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(_gen())
        finally:
            new_loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run_in_new_loop)
        try:
            result = future.result(timeout=30)  # 30秒タイムアウト
            if not result or len(result) < 100:
                raise RuntimeError(f"Edge TTS returned too little data ({len(result)} bytes)")
            return result
        except concurrent.futures.TimeoutError:
            raise RuntimeError("Edge TTS timed out (30s)")

def edge_tts_conv(audio_text, speakers=None):
    """Multi-speaker Edge TTS. Concatenates per-speaker MP3 segments."""
    spk = speakers or ["Man","Woman"]
    fp, mp = EDGE_VF.copy(), EDGE_VM.copy()
    random.shuffle(fp); random.shuffle(mp)
    fi = mi = 0; vm = {}
    for s in spk:
        if is_female(s): vm[s]=fp[fi%len(fp)]; fi+=1
        else: vm[s]=mp[mi%len(mp)]; mi+=1
    print(f"[EDGE-TTS] Voice map: {vm}", flush=True)
    text = audio_text.replace('\\n', '\n')
    parts_raw = re.split(r'(?=\b(?:Man|Woman|Speaker)\s*\d?\s*:)', text, flags=re.I)
    segments = []
    for seg in parts_raw:
        seg = seg.strip()
        if not seg: continue
        m = re.match(r'^((?:Man|Woman|Speaker)\s*\d?)\s*:\s*(.+)', seg, re.I | re.DOTALL)
        if m:
            lbl = m.group(1).strip()
            txt = re.sub(r'^(Man|Woman|Speaker)\s*\d?\s*:\s*', '', m.group(2).strip(), flags=re.I)
            segments.append((lbl, txt))
        else:
            cleaned = re.sub(r'(Man|Woman|Speaker)\s*\d?\s*:\s*', '', seg, flags=re.I).strip()
            if cleaned: segments.append((None, cleaned))
    # Generate per segment + concatenate via ffmpeg
    seg_files = []
    import tempfile, os
    try:
        for lbl, txt in segments:
            if not txt.strip(): continue
            v = vm.get(lbl) if lbl else "en-US-GuyNeural"
            if not v: v = "en-US-AriaNeural" if (lbl and is_female(lbl)) else "en-US-GuyNeural"
            print(f"[EDGE-TTS]   {lbl}: voice={v} text={txt[:50]}", flush=True)
            mp3 = edge_tts_sync(txt, v)
            tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tf.write(mp3); tf.close()
            seg_files.append(tf.name)
        # Concatenate with ffmpeg (insert 250ms silence between speakers)
        if not seg_files: raise RuntimeError("No segments")
        # Build filter_complex for concat with silence
        list_file = tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False)
        for f in seg_files:
            list_file.write(f"file '{f}'\n")
        list_file.close()
        out_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        out_file.close()
        r = subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',list_file.name,'-c','copy',out_file.name],
                          capture_output=True, timeout=60)
        if r.returncode != 0:
            print(f"[EDGE-TTS] ffmpeg concat failed, returning first segment", flush=True)
            with open(seg_files[0], "rb") as f: return f.read()
        with open(out_file.name, "rb") as f: result = f.read()
        os.unlink(list_file.name); os.unlink(out_file.name)
        return result
    finally:
        for f in seg_files:
            try: os.unlink(f)
            except: pass

def mp3_to_opus(mp3, bitrate='16k'):
    """Convert MP3 bytes to Opus bytes via ffmpeg."""
    try:
        r = subprocess.run(['ffmpeg','-i','pipe:','-c:a','libopus','-b:a',bitrate,'-f','webm','pipe:'],
                          input=mp3, capture_output=True, timeout=30)
        return r.stdout if r.returncode == 0 else None
    except FileNotFoundError: return None

GEMINI_TTS_MODELS = ["gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview"]
GEMINI_VOICES_M = ["Puck","Charon","Fenrir","Orus","Achird","Alnilam","Iapetus"]
GEMINI_VOICES_F = ["Aoede","Zephyr","Kore","Leda","Callirrhoe","Despina","Pulcherrima"]

def gemini_tts_conv(audio_text, api_key, speakers=None):
    """Multi-speaker Gemini TTS. Splits by speaker, generates per-line, concatenates PCM."""
    spk = speakers or ["Man","Woman"]
    unique_spk = list(dict.fromkeys(spk))  # preserve order, dedupe
    # Assign distinct voices
    mpool = GEMINI_VOICES_M[:]; random.shuffle(mpool)
    fpool = GEMINI_VOICES_F[:]; random.shuffle(fpool)
    mi = fi = 0; vm = {}
    for s in unique_spk:
        if is_female(s): vm[s]=fpool[fi%len(fpool)]; fi+=1
        else: vm[s]=mpool[mi%len(mpool)]; mi+=1
    print(f"[GEMINI-TTS-CONV] Voice map: {vm}", flush=True)

    # For 2 speakers, try multi-speaker API first (1 request, best quality)
    if len(unique_spk) <= 2:
        try:
            males = [s for s in unique_spk if not is_female(s)]
            females = [s for s in unique_spk if is_female(s)]
            configs = []
            if males: configs.append({"speaker":males[0],"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":vm[males[0]]}}})
            if females: configs.append({"speaker":females[0],"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":vm[females[0]]}}})
            for model in GEMINI_TTS_MODELS:
                resp = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents":[{"parts":[{"text":audio_text[:5000]}]}],
                          "generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{
                              "multiSpeakerVoiceConfig":{"speakerVoiceConfigs":configs}}}},timeout=60)
                if resp.ok:
                    d = resp.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("inlineData",{}).get("data")
                    if d:
                        print(f"[GEMINI-TTS-CONV] 2-speaker multi-speaker OK ({model})", flush=True)
                        return base64.b64decode(d)
                elif resp.status_code in (429,503):
                    time.sleep(3); continue
            print("[GEMINI-TTS-CONV] Multi-speaker failed, falling back to per-line", flush=True)
        except Exception as e:
            print(f"[GEMINI-TTS-CONV] Multi-speaker error: {e}", flush=True)

    # Per-line generation (3+ speakers, or 2-speaker fallback)
    text = audio_text.replace('\\n', '\n')
    parts_raw = re.split(r'(?=\b(?:Man|Woman|Speaker)\s*\d?\s*:)', text, flags=re.I)
    segments = []
    for seg in parts_raw:
        seg = seg.strip()
        if not seg: continue
        m = re.match(r'^((?:Man|Woman|Speaker)\s*\d?)\s*:\s*(.+)', seg, re.I | re.DOTALL)
        if m:
            lbl = m.group(1).strip()
            txt = m.group(2).strip()
            segments.append((lbl, txt))
        else:
            cleaned = re.sub(r'(Man|Woman|Speaker)\s*\d?\s*:\s*', '', seg, flags=re.I).strip()
            if cleaned: segments.append((None, cleaned))

    # Generate PCM per segment
    SILENCE_BYTES = b'\x00' * (24000 * 2 * 300 // 1000)  # 300ms silence (24kHz, 16-bit, mono)
    pcm_parts = []
    ok = 0
    for lbl, txt in segments:
        if not txt.strip(): continue
        voice = vm.get(lbl, random.choice(GEMINI_VOICES_M + GEMINI_VOICES_F))
        try:
            pcm = gemini_tts_single(txt, api_key, voice)
            pcm_parts.append(pcm)
            pcm_parts.append(SILENCE_BYTES)
            ok += 1
            time.sleep(1)  # gentle throttle
        except Exception as e:
            print(f"[GEMINI-TTS-CONV] Line failed ({lbl}): {e}", flush=True)
    if ok == 0: raise RuntimeError("All lines failed")
    result = b''.join(pcm_parts)
    print(f"[GEMINI-TTS-CONV] {ok}/{len(segments)} lines OK, {len(result)//1024}KB", flush=True)
    return result

def gemini_tts_single(text, api_key, voice, max_retries=2):
    """Single-voice Gemini TTS with retry."""
    for model in GEMINI_TTS_MODELS:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents":[{"parts":[{"text":text[:5000]}]}],"generationConfig":{"responseModalities":["AUDIO"],
                          "speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}}}},timeout=60)
                if resp.status_code in (429,503,500):
                    if attempt < max_retries:
                        time.sleep(2 ** (attempt+1)); continue
                    break
                if not resp.ok: raise RuntimeError(f"TTS {resp.status_code}")
                d = resp.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("inlineData",{}).get("data")
                if d: return base64.b64decode(d)
                break
            except (requests.Timeout, requests.ConnectionError):
                if attempt < max_retries: time.sleep(2); continue
                break
    raise RuntimeError(f"TTS single failed")

def gemini_tts(text, api_key, max_retries=3):
    v = random.choice(["Aoede","Puck","Kore","Charon","Zephyr","Fenrir"])
    for model in GEMINI_TTS_MODELS:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents":[{"parts":[{"text":text[:5000]}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":v}}}}},timeout=60)
                if resp.status_code in (429, 503, 500):
                    if attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        print(f"[TTS] {model} {resp.status_code}, retry in {wait}s ({attempt+1}/{max_retries})", flush=True)
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[TTS] {model} failed after {max_retries} retries, trying next model", flush=True)
                        break  # try next model
                if not resp.ok: raise RuntimeError(f"TTS {resp.status_code}")
                d = resp.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("inlineData",{}).get("data")
                if not d: raise RuntimeError("No audio data")
                return base64.b64decode(d)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    print(f"[TTS] {model} timeout, retry ({attempt+1}/{max_retries})", flush=True)
                    time.sleep(2)
                    continue
                break  # try next model
    raise RuntimeError(f"TTS failed: all models exhausted")

def pcm_to_opus(pcm, bitrate='16k'):
    try:
        r = subprocess.run(['ffmpeg','-f','s16le','-ar','24000','-ac','1','-i','pipe:','-c:a','libopus','-b:a',bitrate,'-f','webm','pipe:'],input=pcm,capture_output=True,timeout=30)
        return r.stdout if r.returncode==0 else None
    except FileNotFoundError: return None

# Gemini Image
def gemini_image(scene, api_key, max_retries=2, size="512"):
    """Generate TOEIC-appropriate photograph.
    Smartphone display: 320px JPEG is sufficient. Compressed to ~5-8KB."""
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={api_key}",
                json={
                    "contents":[{"role":"user","parts":[{"text":f"TOEIC test photograph: {scene}. Realistic, no text."}]}],
                    "generationConfig":{
                        "responseModalities":["TEXT","IMAGE"],
                        "imageConfig":{"imageSize":size}
                    }
                },
                timeout=90
            )
            r.raise_for_status()
            for part in r.json().get("candidates",[{}])[0].get("content",{}).get("parts",[]):
                if "inlineData" in part:
                    m, b = part["inlineData"]["mimeType"], part["inlineData"]["data"]
                    try:
                        from PIL import Image
                        im = Image.open(BytesIO(base64.b64decode(b)))
                        # Smartphone: 320px is plenty for TOEIC photos
                        im.thumbnail((320, 320), Image.LANCZOS)
                        buf = BytesIO()
                        im.save(buf, format='JPEG', quality=60, optimize=True)
                        print(f"[IMG] Generated {im.width}x{im.height}, {len(buf.getvalue())//1024}KB jpg", flush=True)
                        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
                    except ImportError:
                        return f"data:{m};base64,{b}"
            raise ValueError("No image in response")
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries:
                wait = 15 if attempt == 0 else 30  # shorter retries since image is small
                print(f"[IMG] Timeout/ConnError (attempt {attempt+1}/{max_retries+1}), retry in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"[IMG] Failed after {max_retries+1} attempts: {type(e).__name__}", flush=True)
                raise

def _generate_listen_audio(qs, real_part):
    """Generate Edge TTS audio for listen-mode Q&A.
    Adds audio_q (question+choices, Part3/4 only) and audio_ans (answer+explanation) to each question."""
    letters = ["A","B","C","D"]
    voice = random.choice(EDGE_VF + EDGE_VM)
    ok = 0

    def _edge_retry(text, retries=3):
        for attempt in range(retries):
            try:
                mp3 = edge_tts_sync(text, voice)
                if mp3: return mp3
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    print(f"[LISTEN-TTS] Edge failed after {retries} retries: {str(e)[:60]}", flush=True)
        return None

    for qi, q in enumerate(qs.get("questions",[])):
        choices = q.get("choices",[])
        correct = q.get("correct",0)
        if correct >= len(choices): continue
        correct_letter = letters[correct]
        correct_text = strip_label(choices[correct])
        
        # Part 3/4: question + choices audio (not needed for Part 1/2, already in main audio)
        if real_part in ("part3","part3_3p","part4") and not q.get("audio_q"):
            q_text = q.get("question","")
            ch_text = ". ".join(f"({letters[i]}) {strip_label(c)}" for i,c in enumerate(choices))
            full_q = f"{q_text}. {ch_text}"
            mp3 = _edge_retry(full_q)
            if mp3:
                o = mp3_to_opus(mp3, '12k')
                if o: q["audio_q"] = base64.b64encode(o).decode(); ok += 1
        
        # Answer + explanation audio (all parts)
        if not q.get("audio_ans"):
            expl_en = q.get("explanation_en","").strip()
            ans_text = f"The answer is {correct_letter}. {correct_text}"
            if expl_en:
                ans_text += f". {expl_en}"
            mp3 = _edge_retry(ans_text)
            if mp3:
                o = mp3_to_opus(mp3, '12k')
                if o: q["audio_ans"] = base64.b64encode(o).decode(); ok += 1
    
    if ok > 0:
        print(f"[LISTEN-TTS] Generated {ok} Q&A audio clips (Edge)", flush=True)

def generate_one_question(level, actual_part, to, engine, model, url, api_key,
                          do_tts, do_img, tts_eng, is_graphic_mode, idx_seed=0):
    """Generate one question (= one qSet that may contain multiple sub-questions).
    Returns the item dict on success, raises on failure."""
    prompt, ap = build_prompt(level, actual_part, to)
    raw = generate_text(prompt, engine, model, url, api_key)
    parsed = parse_json(raw)
    difficulty = parsed.get("difficulty", 0)  # Extract before normalize_set strips it
    qs = enforce_choice_count(normalize_set(parsed, ap))
    if not qs.get("questions"): raise ValueError("No questions")
    # Consistency check BEFORE shuffle (letters still match LLM output)
    if not check_answer_consistency(qs, actual_part):
        print(f"[RETRY] Answer/explanation mismatch detected, regenerating...", flush=True)
        time.sleep(2)
        raw2 = generate_text(prompt, engine, model, url, api_key)
        parsed2 = parse_json(raw2)
        difficulty2 = parsed2.get("difficulty", 0)
        qs2 = enforce_choice_count(normalize_set(parsed2, ap))
        if qs2.get("questions") and check_answer_consistency(qs2, actual_part):
            qs = qs2
            difficulty = difficulty2
        else:
            print(f"[WARN] Retry also inconsistent, keeping original", flush=True)
    # Shuffle AFTER consistency check
    qs = shuffle_answer_positions(qs)
    qs["_questionType"] = to.get("type", "unknown")  # Save type for diagnosis
    real_part = qs.get("part", actual_part)
    item = {"part": real_part, "level": level,
            "createdAt": int(time.time()*1000) + idx_seed,
            "difficulty": difficulty,
            "qSet": qs, "imgUrl": None, "audioOpus": None}
    # TTS
    if do_tts and qs.get("audio"):
        try:
            at = qs["audio"]
            if tts_eng == "edge" and check_edge_tts():
                print(f"[TTS] Part={real_part}, engine=edge{'_conv' if real_part=='part3' else ''}", flush=True)
                mp3 = edge_tts_conv(at, qs.get("speakers")) if real_part in ("part3","part3_3p") else edge_tts_sync(at, random.choice(EDGE_VF+EDGE_VM))
                o = mp3_to_opus(mp3)
                if o:
                    item["audioOpus"] = base64.b64encode(o).decode(); item["audioFormat"] = "opus"
                    print(f"[TTS] ✅ OK ({len(o)//1024}KB opus)", flush=True)
                else:
                    print(f"[TTS] ⚠️ Edge TTS: mp3→opus conversion failed", flush=True)
            elif tts_eng == "azure" and st.session_state.get("azure_speech_key"):
                az_key = st.session_state.azure_speech_key
                az_region = st.session_state.get("azure_speech_region", "eastus")
                print(f"[TTS] Part={real_part}, engine=azure{'_conv' if real_part=='part3' else ''}", flush=True)
                mp3 = azure_tts_conv(at, az_key, az_region, qs.get("speakers")) if real_part in ("part3","part3_3p") else azure_tts(at, az_key, az_region)
                o = mp3_to_opus(mp3)
                if o:
                    item["audioOpus"] = base64.b64encode(o).decode(); item["audioFormat"] = "opus"
                    print(f"[TTS] ✅ OK ({len(o)//1024}KB opus)", flush=True)
                else:
                    print(f"[TTS] ⚠️ Azure TTS: mp3→opus conversion failed", flush=True)
            elif tts_eng == "gemini" and api_key:
                if real_part in ("part3","part3_3p"):
                    print(f"[TTS] Part={real_part}, engine=gemini_conv ({len(qs.get('speakers',[]))} speakers)", flush=True)
                    p = gemini_tts_conv(at, api_key, qs.get("speakers")); o = pcm_to_opus(p)
                else:
                    print(f"[TTS] Part={real_part}, engine=gemini ({len(at)}c)", flush=True)
                    p = gemini_tts(at, api_key); o = pcm_to_opus(p)
                if o:
                    item["audioOpus"] = base64.b64encode(o).decode(); item["audioFormat"] = "opus"
                    print(f"[TTS] ✅ OK ({len(o)//1024}KB opus)", flush=True)
                else:
                    print(f"[TTS] ⚠️ Gemini TTS: pcm→opus conversion failed", flush=True)
                time.sleep(6.5)
            else:
                print(f"[TTS] ⏭️ skipped (engine={tts_eng}, key={'set' if api_key else 'none'})", flush=True)
        except Exception as e:
            print(f"[TTS] ❌ {e}", flush=True)
    elif do_tts and not qs.get("audio"):
        print(f"[TTS] ⏭️ no audio field in qSet", flush=True)
    # Image
    if do_img and api_key:
        if qs.get("scene"):  # Part 1
            try: item["imgUrl"] = gemini_image(qs["scene"], api_key); time.sleep(5)  # 15 RPM on Tier 1 = 4s min, 5s for safety
            except Exception as e: print(f"[IMG] Part1 error: {e}", flush=True)
        # Graphic images are REQUIRED for maps, floor plans, charts etc.
        # HTML imports and displays them. Tables alone are insufficient for visual content.
        elif is_graphic_mode and qs.get("graphic"):
            g = qs["graphic"]
            desc = f"Clean business graphic for a test: {g.get('title','')}. "
            if g.get("headers") and g.get("rows"):
                desc += f"A table with columns: {', '.join(g['headers'])}. "
                desc += f"Rows: {'; '.join([', '.join(str(c) for c in r) for r in g['rows'][:3]])}. "
            desc += "Simple, professional style. No decorations. Clear text. White background."
            try:
                print(f"[IMG] Generating graphic: {desc[:80]}...", flush=True)
                item["imgUrl"] = gemini_image(desc, api_key, size="512")  # 512px for all (TOEIC images are small)
                time.sleep(5)
            except Exception as e:
                print(f"[IMG] Graphic error: {e}", flush=True)
    # Vocab audio - Edge preferred (free, avoids Gemini 429), Gemini as fallback
    if qs.get("vocab") and tts_eng != "off":
        use_edge_for_vocab = check_edge_tts()
        def _try_edge(text):
            for attempt in range(2):
                try:
                    mp3 = edge_tts_sync(text, random.choice(EDGE_VF+EDGE_VM))
                    if mp3: return mp3
                except Exception as e:
                    if attempt == 0:
                        print(f"[VOCAB] Edge retry for '{text[:30]}': {e}", flush=True)
                        time.sleep(0.5)
            return None
        w_ok, e_ok = 0, 0
        for vw in qs["vocab"]:
            word = vw.get("word", "")
            ex = vw.get("example", "")
            if not word: continue
            try:
                if use_edge_for_vocab:
                    mp3w = _try_edge(word)
                    if mp3w:
                        oo = mp3_to_opus(mp3w, '12k')
                        if oo: vw["audio"] = base64.b64encode(oo).decode(); w_ok += 1
                    if ex:
                        mp3e = _try_edge(ex)
                        if mp3e:
                            eo = mp3_to_opus(mp3e, '12k')
                            if eo: vw["example_audio"] = base64.b64encode(eo).decode(); e_ok += 1
                elif tts_eng == "gemini" and api_key:
                    pp = gemini_tts(word, api_key); oo = pcm_to_opus(pp, '12k')
                    if oo: vw["audio"] = base64.b64encode(oo).decode(); w_ok += 1
                    time.sleep(4)
                    if ex:
                        ep = gemini_tts(ex, api_key); eo = pcm_to_opus(ep, '12k')
                        if eo: vw["example_audio"] = base64.b64encode(eo).decode(); e_ok += 1
                        time.sleep(4)
            except Exception as e:
                print(f"[VOCAB] {e}", flush=True)
        print(f"[VOCAB] Audio: {w_ok} words, {e_ok} examples ({('Edge' if use_edge_for_vocab else 'Gemini')})", flush=True)
    # Listen-mode Q&A audio (Edge TTS — Part 2/3/4 only. Part 1 is photo-based, not suited for listening flow)
    if real_part in ("part2","part3","part3_3p","part4") and qs.get("questions") and tts_eng != "off" and check_edge_tts():
        _generate_listen_audio(qs, real_part)
    # Validate: Listening parts REQUIRE audio. If TTS was enabled but audio missing, mark invalid.
    is_listening_part = real_part in ("part1","part2","part3","part3_3p","part4")
    if is_listening_part and do_tts and not item.get("audioOpus"):
        item["_invalid"] = "listening_no_audio"
        print(f"[VALIDATE] Part={real_part}: SKIP (no audio generated)", flush=True)
    # Store validation-relevant flags BEFORE stripping
    item["_hasAudio"] = bool(item.get("audioOpus"))
    item["_hasImage"] = bool(item.get("imgUrl"))
    _strip_audio(item)  # Move audio to _audio_store for lightweight session_state
    return item


def validate_stock_item(item, require_tts=True, require_image_for_part1=True, require_image_for_graphic=False, require_vocab_audio=False, strict_vocab=False):
    """
    Unified validation before saving to stock.
    Returns (is_valid: bool, reason: str | None).

    Rules (hard - fails validation):
    - Listening parts (1-4) MUST have audioOpus (if TTS was requested)
    - Part 1 MUST have imgUrl (if image was requested)
    - Graphic questions MUST have imgUrl (if image was requested)
    - Questions list must be non-empty

    Rules (soft - warn only unless strict_vocab=True):
    - vocab word audio: strongly preferred but not required (can fallback at playback)
    - vocab example audio: optional (shorter fallback is fine)
    """
    if not item or not isinstance(item, dict):
        return False, "not a dict"
    part = item.get("part", "")
    qs = item.get("qSet", {})
    if not qs.get("questions"):
        return False, "no questions"
    # Listening audio (check both direct field and flag set before stripping)
    is_listening = part in ("part1","part2","part3","part3_3p","part4")
    if is_listening and require_tts and not item.get("audioOpus") and not item.get("_hasAudio"):
        return False, "listening without audio"
    # Part 1 image
    if part == "part1" and require_image_for_part1 and not item.get("imgUrl") and not item.get("_hasImage"):
        return False, "part1 without image"
    # Graphic questions: need BOTH image AND table data.
    # Maps, floor plans, charts can't be rendered as text tables — image is mandatory.
    qs_obj = item.get("qSet", {})
    graphic = qs_obj.get("graphic")
    if graphic:
        headers = graphic.get("headers") if isinstance(graphic, dict) else None
        rows = graphic.get("rows") if isinstance(graphic, dict) else None
        if not headers or not rows:
            return False, "graphic missing headers/rows"
        if require_image_for_graphic and not item.get("imgUrl"):
            return False, "graphic without image"
    # Vocab audio check (soft by default, strict only if strict_vocab=True)
    vocab = qs_obj.get("vocab", [])
    if vocab and require_vocab_audio:
        missing_word = sum(1 for v in vocab if not v.get("audio"))
        missing_example = sum(1 for v in vocab if v.get("example") and not v.get("example_audio"))
        if strict_vocab:
            # Hard fail if any vocab item missing audio
            for i, v in enumerate(vocab):
                if not v.get("audio"):
                    return False, f"vocab[{i}] missing word audio"
                if v.get("example") and not v.get("example_audio"):
                    return False, f"vocab[{i}] missing example audio"
        else:
            # Warn only (vocab audio can be regenerated at playback)
            if missing_word > 0 or missing_example > 0:
                print(f"[VALIDATE] vocab audio incomplete (words:{len(vocab)-missing_word}/{len(vocab)}, examples:{missing_example} missing) - saving anyway", flush=True)
    return True, None


# ══════════════════════════════════════
# Mock Test Composition
# ══════════════════════════════════════
# 本番TOEICのパート別問題数（フル200問 — Part 3/7はサブタイプ別に管理）
MOCK_FULL_DIST = {
    "part1": 6, "part2": 25,
    "part3": 33,     # 2人会話: 11セット × 3問
    "part3_3p": 6,   # 3人会話: 2セット × 3問 (本番は13セット中1-2が3人)
    "part4": 30,
    "part5": 30, "part6": 16,
    "part7s": 29,  # Single passage
    "part7d": 10,  # Double passage
    "part7t": 15,  # Triple passage
}
# 1セット(qSet)あたりの問題数 → 必要セット数を計算
QS_PER_SET = {
    "part1":1, "part2":1, "part3":3, "part3_3p":3, "part4":3,
    "part5":1, "part6":4,
    "part7s":3, "part7d":5, "part7t":5,
}

# 本番TOEIC graphic問題セット数 (フル模試基準)
MOCK_GRAPHIC_SETS_FULL = {"part3": 3, "part4": 3, "part7s": 2}
MOCK_GRAPHIC_SETS_HALF = {"part3": 2, "part4": 1, "part7s": 1}

# 本番想定の難易度ミックス
DIFFICULTY_MIX = {
    "part1": [("beginner",0.6),("intermediate",0.3),("advanced",0.1)],
    "part2": [("beginner",0.3),("intermediate",0.5),("advanced",0.2)],
    "part3": [("beginner",0.2),("intermediate",0.5),("advanced",0.3)],
    "part3_3p": [("beginner",0.1),("intermediate",0.5),("advanced",0.4)],
    "part4": [("beginner",0.2),("intermediate",0.5),("advanced",0.3)],
    "part5": [("beginner",0.3),("intermediate",0.5),("advanced",0.2)],
    "part6": [("beginner",0.2),("intermediate",0.5),("advanced",0.3)],
    "part7s": [("beginner",0.2),("intermediate",0.5),("advanced",0.3)],
    "part7d": [("beginner",0.2),("intermediate",0.5),("advanced",0.3)],
    "part7t": [("beginner",0.1),("intermediate",0.5),("advanced",0.4)],
}

def build_mock_plan(scale=1.0):
    """Build a list of (part, level) tuples to generate.
    scale=1.0 → full 200q test, scale=0.5 → half 100q test."""
    plan = []
    for part, total_q in MOCK_FULL_DIST.items():
        target_q = max(1, round(total_q * scale))
        sets_needed = max(1, -(-target_q // QS_PER_SET[part]))  # ceil division
        # Distribute sets across difficulty levels
        mix = DIFFICULTY_MIX[part]
        levels_assigned = []
        for level, ratio in mix:
            n = round(sets_needed * ratio)
            levels_assigned.extend([level] * n)
        # Adjust if rounding made total off
        while len(levels_assigned) < sets_needed:
            levels_assigned.append(mix[1][0])  # add intermediate
        levels_assigned = levels_assigned[:sets_needed]
        random.shuffle(levels_assigned)
        for lv in levels_assigned:
            plan.append((part, lv))
    return plan


# ══════════════════════════════════════
# Persistence (auto-save to local JSON file)
# ══════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_FILE = SCRIPT_DIR / "results.json"
MOCK_DIR = SCRIPT_DIR / "mock_data"  # Each batch = separate file

# ── Audio store: separate from session_state for performance ──
# session_state holds lightweight items (no audioOpus)
# _audio_store holds audioOpus keyed by createdAt
# Use cache_resource to persist across reruns (not serialized with session_state)
@st.cache_resource
def _get_audio_store():
    return {}
_audio_store = _get_audio_store()

def _strip_audio(item):
    """Remove ALL audio from item, store in _audio_store. Returns lightweight item."""
    ts = item.get("createdAt", 0)
    audio_data = {}
    # Main audio
    if item.get("audioOpus"):
        audio_data["audioOpus"] = item.pop("audioOpus")
        item.pop("audioFormat", None)
    # Vocab audio
    qs = item.get("qSet", {})
    for vi, v in enumerate(qs.get("vocab", [])):
        if v.get("audio"):
            audio_data[f"v{vi}_a"] = v.pop("audio")
        if v.get("example_audio"):
            audio_data[f"v{vi}_e"] = v.pop("example_audio")
    # Listen Q&A audio
    for qi, q in enumerate(qs.get("questions", [])):
        if q.get("audio_q"):
            audio_data[f"q{qi}_q"] = q.pop("audio_q")
        if q.get("audio_ans"):
            audio_data[f"q{qi}_a"] = q.pop("audio_ans")
    if audio_data:
        _audio_store[ts] = audio_data
    return item

def _restore_audio(item):
    """Restore ALL audio from _audio_store into item for save/export."""
    ts = item.get("createdAt", 0)
    ad = _audio_store.get(ts, {})
    if not ad: return item
    if ad.get("audioOpus"):
        item["audioOpus"] = ad["audioOpus"]
        item["audioFormat"] = "opus"
    qs = item.get("qSet", {})
    for vi, v in enumerate(qs.get("vocab", [])):
        if ad.get(f"v{vi}_a"): v["audio"] = ad[f"v{vi}_a"]
        if ad.get(f"v{vi}_e"): v["example_audio"] = ad[f"v{vi}_e"]
    for qi, q in enumerate(qs.get("questions", [])):
        if ad.get(f"q{qi}_q"): q["audio_q"] = ad[f"q{qi}_q"]
        if ad.get(f"q{qi}_a"): q["audio_ans"] = ad[f"q{qi}_a"]
    return item

def get_audio(item):
    """Get main audioOpus for an item."""
    ts = item.get("createdAt", 0)
    ad = _audio_store.get(ts, {})
    return ad.get("audioOpus") or item.get("audioOpus")

def load_results(filepath):
    """Load results from JSON file. Strips audioOpus into _audio_store for lightweight session_state."""
    global _audio_store
    try:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    cleaned = []
                    purged = 0
                    audio_count = 0
                    for item in data:
                        qs = item.get("qSet", {})
                        if not item.get("part") or not qs.get("questions"):
                            purged += 1
                            continue
                        if item.get("audioOpus"):
                            audio_count += 1
                        _strip_audio(item)
                        cleaned.append(item)
                    if purged > 0:
                        try:
                            # Save cleaned full version (with audio restored)
                            full = [_restore_audio(dict(**i)) for i in cleaned]
                            with open(filepath, "w", encoding="utf-8") as fw:
                                json.dump(full, fw, ensure_ascii=False)
                            # Re-strip after save
                            for i in cleaned: _strip_audio(i)
                        except Exception: pass
                    print(f"[PERSIST] Loaded {len(cleaned)} items from {filepath.name} ({audio_count} with audio, {len(_audio_store)} in audio_store)" + (f" (purged {purged})" if purged else ""), flush=True)
                    return cleaned
    except Exception as e:
        print(f"[PERSIST] Load error {filepath.name}: {e}", flush=True)
    return []

def save_results(filepath, items):
    """Save results to JSON file. Merges audioOpus back from _audio_store."""
    try:
        # Reconstruct full items with audio for saving
        full_items = []
        for item in items:
            full = json.loads(json.dumps(item, ensure_ascii=False))  # deep copy
            _restore_audio(full)
            full_items.append(full)
        tmp = filepath.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(full_items, f, ensure_ascii=False)
        tmp.replace(filepath)
    except Exception as e:
        print(f"[PERSIST] Save error {filepath.name}: {e}", flush=True)


# ── Mock folder persistence ──
def _ensure_mock_dir():
    MOCK_DIR.mkdir(exist_ok=True)

def _mock_batch_path(batch_id):
    return MOCK_DIR / f"mock_{batch_id}.json"

def save_mock_batch(batch_id, items):
    """Save a single batch to its own file in mock_data/."""
    _ensure_mock_dir()
    batch_items = [r for r in items if r.get("batchId") == batch_id]
    if not batch_items:
        return
    fp = _mock_batch_path(batch_id)
    try:
        # Restore audio from _audio_store for saving
        full_items = []
        for item in batch_items:
            full = json.loads(json.dumps(item, ensure_ascii=False))
            _restore_audio(full)
            full_items.append(full)
        tmp = fp.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(full_items, f, ensure_ascii=False)
        tmp.replace(fp)
        print(f"[MOCK] Saved batch {batch_id} → {fp.name} ({len(batch_items)} items)", flush=True)
    except Exception as e:
        print(f"[MOCK] Save error: {e}", flush=True)

def delete_mock_batch(batch_id):
    """Delete a batch file."""
    fp = _mock_batch_path(batch_id)
    if fp.exists():
        fp.unlink()
        print(f"[MOCK] Deleted {fp.name}", flush=True)

def load_all_mock_batches():
    """Load all batch files from mock_data/ folder."""
    _ensure_mock_dir()
    all_items = []
    # Also migrate old single-file format if it exists
    old_file = SCRIPT_DIR / "mock_results.json"
    if old_file.exists():
        try:
            with open(old_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if isinstance(old_data, list) and len(old_data) > 0:
                # Group by batchId and save as separate files
                batches = {}
                for item in old_data:
                    bid = item.get("batchId", "legacy")
                    if bid not in batches:
                        batches[bid] = []
                    batches[bid].append(item)
                for bid, items in batches.items():
                    fp = _mock_batch_path(bid)
                    if not fp.exists():
                        with open(fp, "w", encoding="utf-8") as f:
                            json.dump(items, f, ensure_ascii=False)
                        print(f"[MOCK] Migrated batch {bid} ({len(items)} items) from old file", flush=True)
                # Rename old file to backup
                old_file.rename(old_file.with_suffix(".json.bak"))
                print(f"[MOCK] Old mock_results.json → .bak", flush=True)
        except Exception as e:
            print(f"[MOCK] Migration error: {e}", flush=True)
    # Load all batch files
    for fp in sorted(MOCK_DIR.glob("mock_*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                valid = []
                for item in data:
                    ok, reason = validate_stock_item(item, require_tts=False, require_image_for_part1=False,
                                                     require_image_for_graphic=False, require_vocab_audio=False)
                    if ok:
                        _strip_audio(item)
                        valid.append(item)
                all_items.extend(valid)
                print(f"[MOCK] Loaded {fp.name}: {len(valid)} items", flush=True)
        except Exception as e:
            print(f"[MOCK] Error loading {fp.name}: {e}", flush=True)
    return all_items

def clear_all_mock_batches():
    """Delete all batch files."""
    _ensure_mock_dir()
    for fp in MOCK_DIR.glob("mock_*.json"):
        fp.unlink()
    print("[MOCK] All batches cleared", flush=True)


if "_init" not in st.session_state:
    # Set ALL defaults FIRST (before any logic that might fail)
    st.session_state.ollama_url = os.environ.get("OLLAMA_URL","http://localhost:11434")
    st.session_state.gemini_key = os.environ.get("GEMINI_API_KEY","")
    st.session_state.azure_speech_key = os.environ.get("AZURE_SPEECH_KEY","")
    st.session_state.azure_speech_region = os.environ.get("AZURE_SPEECH_REGION","eastus")
    st.session_state.azure_speech_endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT","")
    st.session_state.model_key = "auto (per-part recommended)"
    st.session_state.part = "part5"; st.session_state.level = "advanced"; st.session_state.count = 10
    st.session_state.enable_tts = True; st.session_state.enable_image = False
    st.session_state.prac_idx = 0; st.session_state.prac_answered = {}
    # Default TTS: Azure if key set, else Edge if available, else Gemini
    try:
        _default_tts = "azure" if st.session_state.azure_speech_key else ("edge" if check_edge_tts() else "gemini")
    except Exception:
        _default_tts = "gemini"
    st.session_state.tts_engine = _default_tts
    # Load persisted data on startup
    st.session_state.results = load_results(RESULTS_FILE)
    st.session_state.mock_results = load_all_mock_batches()
    gk = st.session_state.gemini_key
    print(f"[INIT] key={'set' if gk else 'empty'} | results={len(st.session_state.results)} | mock={len(st.session_state.mock_results)}", flush=True)
    st.session_state._init = True  # Set LAST so incomplete init retries

# Safety defaults — ensure critical keys exist even if init was from old version
for _k, _v in {"enable_tts": False, "enable_image": False, "prac_idx": 0, "prac_answered": {},
               "tts_engine": "edge", "azure_speech_key": "", "azure_speech_region": "eastus",
               "azure_speech_endpoint": ""}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════
# Sidebar
# ══════════════════════════════════════
# ══════════════════════════════════════
# Vocab helper functions (module level for reuse + caching)
# ══════════════════════════════════════
def lemmatize(w):
    """Basic English lemmatization to dictionary form."""
    w = w.lower().strip()
    if w.endswith("ied") and len(w)>4: return w[:-3]+"y"
    if w.endswith("ing") and len(w)>5 and w[-4]==w[-5]: return w[:-4]
    if w.endswith("ting") and len(w)>5: return w[:-4]+"te"
    if w.endswith("ing") and len(w)>4: return w[:-3]
    if w.endswith("ed") and len(w)>4 and w[-3]==w[-4]: return w[:-3]
    if w.endswith("ed") and len(w)>3: return w[:-2]
    if w.endswith("es") and len(w)>4: return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w)>3: return w[:-1]
    return w

def norm_meaning(m):
    if not m: return ""
    return re.split(r'[、,（(]', m.strip())[0].strip()

def meaning_segments(m):
    if not m: return set()
    # Split by all bracket types, commas, and ・
    return set(s.strip() for s in re.split(r'[、,（()）)・「」【】]', m.strip()) if s.strip())

def _ja_core(s):
    """Extract core meaning by stripping particles, prefixes, suffixes."""
    s = s.strip()
    s = s.lstrip("～〜を")
    # Remove trailing particles/copula
    s = re.sub(r'[をにがでとのはもな]+$', '', s)
    # Normalize verb/adj endings
    s = re.sub(r'(する|して|した|している|させる|される|しない|すること|するもの)$', 'する', s)
    s = re.sub(r'(いる|いて|いた|いない)$', 'いる', s)
    s = re.sub(r'(れる|れて|れた)$', 'れる', s)
    s = re.sub(r'(ある|あって|あった)$', 'ある', s)
    # Remove common adj suffixes: 的な→的, のある→""
    s = re.sub(r'的な$', '的', s)
    s = re.sub(r'のある$', '', s)
    return s

def meanings_match(m1, m2):
    """Check if two Japanese meanings are semantically similar."""
    if not m1 or not m2: return False
    if m1.strip() == m2.strip(): return True
    
    s1 = meaning_segments(m1)
    s2 = meaning_segments(m2)
    # Exact segment overlap
    if s1 & s2: return True
    
    # Core-normalized overlap
    c1 = {_ja_core(s) for s in s1 if len(s) >= 1}
    c2 = {_ja_core(s) for s in s2 if len(s) >= 1}
    c1.discard(""); c2.discard("")
    if c1 & c2: return True
    
    # Substring containment (min 2 chars for Japanese)
    for a in c1:
        for b in c2:
            if len(a) >= 2 and len(b) >= 2:
                if a in b or b in a: return True
    
    # Kanji root extraction: "迅速に処理する" → {"迅速", "処理"}
    kanji_re = re.compile(r'[\u4e00-\u9fff]{2,}')
    k1 = set()
    k2 = set()
    for s in s1: k1.update(kanji_re.findall(s))
    for s in s2: k2.update(kanji_re.findall(s))
    if k1 and k2:
        if k1 & k2: return True
        # Kanji substring: "迅速" ⊂ "迅速化"
        for a in k1:
            for b in k2:
                if len(a) >= 2 and len(b) >= 2 and (a in b or b in a): return True
    
    return False


def _do_llm_vocab_cleanup(all_vocab):
    """Use LLM to deduplicate similar meanings for each word."""
    multi = [(v["word"], v.get("_meanings",[])) for v in all_vocab if len(v.get("_meanings",[])) >= 2]
    if not multi:
        st.info("重複候補の単語がありません")
        return

    url = st.session_state.ollama_url
    api_key = st.session_state.gemini_key
    prog = st.progress(0)
    stat = st.empty()
    stat.info(f"🤖 {len(multi)}語の意味を AI で整理中...")

    BATCH = 10
    keep_map = {}
    for bi in range(0, len(multi), BATCH):
        batch = multi[bi:bi+BATCH]
        prompt_lines = []
        for word, meanings in batch:
            m_list = " / ".join(m.get('ja','') for m in meanings)
            prompt_lines.append(f"  {word}: {m_list}")
        prompt = f"""英単語の日本語訳を整理する。同じ意味の重複を統合し、本質的に異なる意味だけ残せ。

判定基準:
- 「課す」「課する」→ 同じ（活用違い）→ 1つにまとめる
- 「迅速に処理する」「迅速化する」「早める」→ 同じ概念 → 最も自然な1つ
- 「売却、分離」「事業売却」→ 同じ（売却が共通）→ 1つ
- 「独占的な」「独占の」→ 同じ（語尾違い）→ 1つ
- 「連絡係」「連携」→ 異なる → 両方残す

具体例:
INPUT: expedite: を迅速に処理する / を迅速化する / を早める / をはかどらせる
OUTPUT: {{"expedite": ["迅速に処理する"]}}

INPUT: levy: （税金を）課す / （税金を）課する
OUTPUT: {{"levy": ["（税金を）課す"]}}

INPUT: liaison: 連絡係、窓口 / 連絡担当者 / 連絡、連携
OUTPUT: {{"liaison": ["連絡係", "連携"]}}

以下を整理せよ:
{chr(10).join(prompt_lines)}

JSONのみ出力:"""

        try:
            try:
                resp = requests.post(f"{url}/api/generate", json={
                    "model":"gemma3:12b","prompt":prompt,"stream":False,
                    "options":{"temperature":0.1,"num_predict":2048,"num_gpu":99}
                }, timeout=120)
                if resp.ok:
                    raw = resp.json().get("response","")
                else:
                    raise RuntimeError(f"Ollama {resp.status_code}")
            except Exception:
                if not api_key: raise
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
                    json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.1}},
                    timeout=60)
                if not resp.ok: raise RuntimeError(f"Gemini {resp.status_code}")
                raw = resp.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")

            raw = re.sub(r'```json\s*|```\s*', '', raw).strip()
            parsed = json.loads(raw)
            for word, kept in parsed.items():
                if isinstance(kept, list):
                    keep_map[word.lower().strip()] = [str(k) for k in kept]
            print(f"[VOCAB-AI] Batch {bi//BATCH+1}: {len(parsed)} words processed", flush=True)
        except Exception as e:
            print(f"[VOCAB-AI] Batch error: {e}", flush=True)
        prog.progress(min(1.0, (bi+BATCH)/len(multi)))

    if not keep_map:
        stat.warning("⚠️ AI整理に失敗しました")
        return

    # Apply: remove meanings not in keep_map
    cleaned = 0
    for r in st.session_state.results:
        qs = r.get("qSet", {})
        vocab = qs.get("vocab", [])
        if not vocab: continue
        new_vocab = []
        for v in vocab:
            word = v.get("word","").strip()
            ja = v.get("ja","").strip()
            base = word.lower().strip()
            if base in keep_map:
                kept_list = keep_map[base]
                # Check if this ja matches any kept meaning
                is_kept = any(meanings_match(ja, k) or ja in k or k in ja for k in kept_list)
                if not is_kept:
                    cleaned += 1
                    continue
            new_vocab.append(v)
        qs["vocab"] = new_vocab

    save_results(RESULTS_FILE, st.session_state.results)
    stat.success(f"✅ AI整理完了: {cleaned}件の重複意味を削除 ({len(keep_map)}語を分析)")
    st.rerun()

def build_vocab_list(results):
    """Build merged vocab list from results. Restores audio from _audio_store."""
    all_vocab = []
    word_map = {}
    for r in results:
        qs = r.get("qSet", {})
        part = r.get("part","?")
        level = r.get("level","?")
        ts = r.get("createdAt", 0)
        ad = _audio_store.get(ts, {})  # Restore stripped audio
        for vi, v in enumerate(qs.get("vocab", [])):
            word = v.get("word","").strip()
            ja = v.get("ja","").strip()
            example = v.get("example","").strip()
            pos = v.get("pos","other").strip().lower()
            audio = v.get("audio","") or ad.get(f"v{vi}_a","")
            ex_audio = v.get("example_audio","") or ad.get(f"v{vi}_e","")
            if not word: continue
            word_level = v.get("level", "").strip().upper()  # A/B/C word difficulty
            if word_level not in ("A","B","C"): word_level = ""
            base = lemmatize(word)
            if base in word_map:
                idx = word_map[base]
                existing_meanings = all_vocab[idx].get("_meanings",[])
                has_overlap = any(meanings_match(m["ja"], ja) for m in existing_meanings)
                if ja and not has_overlap:
                    existing_meanings.append({"ja":ja,"example":example,"example_audio":ex_audio})
                    all_vocab[idx]["_meanings"] = existing_meanings
                if audio and not all_vocab[idx].get("_audio"):
                    all_vocab[idx]["_audio"] = audio
                if word_level and not all_vocab[idx].get("_word_level"):
                    all_vocab[idx]["_word_level"] = word_level
            else:
                entry = {"word":word, "ja":ja, "example":example, "_part":part, "_level":level,
                         "_pos":pos, "_audio":audio, "_example_audio":ex_audio, "_word_level":word_level,
                         "_meanings":[{"ja":ja,"example":example,"example_audio":ex_audio}]}
                word_map[base] = len(all_vocab)
                all_vocab.append(entry)
    return all_vocab

# ══════════════════════════════════════
# Main — Tabs: Generate / Practice
# ══════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Connection")
    st.text_input("Ollama URL", key="ollama_url")
    st.text_input("Gemini API Key", key="gemini_key", type="password")

    st.divider()
    st.markdown("## 🤖 Model")
    valid = ["auto (per-part recommended)"] + [l for l in MODEL_OPTIONS if MODEL_OPTIONS[l] is not None]
    st.selectbox("Model", valid, key="model_key")
    sel = MODEL_OPTIONS.get(st.session_state.model_key)
    if st.session_state.model_key.startswith("auto"):
        st.caption("🔄 Auto: Part 2/4/5 → gemma3:12b, Part 1/3/6/7 → gemini-2.5-flash")
    elif sel:
        tag = "🖥️ Local" if sel["engine"]=="ollama" else "☁️ Cloud"
        st.caption(f"{tag} `{sel['model']}`")

    if st.button("🔌 Test"):
        if st.session_state.model_key.startswith("auto"):
            # Test both Ollama and Gemini
            try:
                r = requests.get(f"{st.session_state.ollama_url}/api/tags",timeout=5)
                st.success(f"✅ Ollama: {', '.join(m['name'] for m in r.json().get('models',[])[:5])}")
            except Exception as e: st.error(f"❌ Ollama: {e}")
            if st.session_state.gemini_key:
                st.success("✅ Gemini API key set")
            else:
                st.warning("⚠️ Gemini API key missing (needed for Part 1/3/6/7)")
        elif sel and sel["engine"]=="ollama":
            try:
                r = requests.get(f"{st.session_state.ollama_url}/api/tags",timeout=5)
                st.success(f"✅ {', '.join(m['name'] for m in r.json().get('models',[])[:5])}")
            except Exception as e: st.error(f"❌ {e}")
        else:
            st.success("✅ Key set" if st.session_state.gemini_key else "❌ No key")

    st.divider()
    st.markdown("## 🔊 TTS")
    edge = check_edge_tts()
    opts = []
    if edge: opts.append("edge")
    opts += ["azure", "gemini", "off"]
    if st.session_state.tts_engine not in opts:
        st.session_state.tts_engine = opts[0]
    st.radio("Engine", opts, key="tts_engine",
             format_func={"edge":"Edge TTS (無料)","azure":"☁️ Azure Speech ($15/1M文字)","gemini":"🌟 Gemini TTS","off":"🔇 Off"}.get,
             horizontal=True)
    if st.session_state.tts_engine == "azure":
        st.text_input("Azure Speech Key", key="azure_speech_key", type="password")
        st.text_input("Azure Endpoint", key="azure_speech_endpoint",
                       help="例: https://toeic-tts.cognitiveservices.azure.com")
        if not st.session_state.get("azure_speech_endpoint"):
            rgn = st.session_state.get("azure_speech_region","eastus")
            st.caption(f"⚠️ Endpoint未設定 → リージョン `{rgn}` を使用")
    if not edge: st.caption("⚠️ Edge TTS が使えません: `pip install edge-tts`")

    st.divider()
    st.caption("v2026.04.28e · level-aware difficulty rating (vocab/quiz/listen removed) · 303 types")

st.markdown("<h1 style='text-align:center;background:linear-gradient(135deg,#818cf8,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:28px'>📝 TOEIC Generator</h1>", unsafe_allow_html=True)

tab_gen, tab_manage, tab_practice, tab_mock_test = st.tabs(["🔧 Generate", "📦 管理", "🎯 Practice", "📝 模試テスト"])

# ══════════════════════════════════════
# TAB: 管理 (Management)
# ══════════════════════════════════════
with tab_manage:
    st.divider()
    st.markdown(f"**Results: {len(st.session_state.results)}**")
    if st.session_state.results:
        # Quick summary (no expander needed)
        by_part = {}
        for r in st.session_state.results:
            p = r.get("part","?")
            by_part[p] = by_part.get(p, 0) + 1
        part_labels = {"part1":"P1","part2":"P2","part3":"P3","part4":"P4","part5":"P5","part6":"P6","part7":"P7"}
        summary = " / ".join(f"{part_labels.get(p,p)}:{n}" for p,n in sorted(by_part.items()))
        st.caption(summary)

        # Export/Delete
        with st.expander("📦 エクスポート / 🗑️ 削除"):
            # Detailed breakdown
            level_counts = {}
            for r in st.session_state.results:
                p = r.get("part","?")
                lv = r.get("level","?")
                level_counts[(p,lv)] = level_counts.get((p,lv),0) + 1

            filter_options = ["全パート"] + sorted(by_part.keys())
            selected_part = st.selectbox("パート選択", filter_options,
                format_func=lambda x: f"全パート ({len(st.session_state.results)})" if x=="全パート" else f"{part_labels.get(x,x)} ({by_part.get(x,0)}問)",
                key="stock_part_filter")

            if selected_part == "全パート":
                lv_info = {}
                for (p,lv), c in level_counts.items():
                    lv_info[lv] = lv_info.get(lv,0) + c
            else:
                lv_info = {}
                for (p,lv), c in level_counts.items():
                    if p == selected_part:
                        lv_info[lv] = lv_info.get(lv,0) + c
            if lv_info:
                lv_str = " / ".join(f"{lv}:{n}" for lv,n in sorted(lv_info.items()))
                st.caption(f"レベル内訳: {lv_str}")

            # --- Export: full + differential ---
            LAST_EXPORT_FILE = "last_html_export.txt"
            last_export_ts = 0
            try:
                with open(LAST_EXPORT_FILE) as f:
                    last_export_ts = int(f.read().strip())
            except Exception:
                pass

            try:
                with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                    _exp_all = json.load(f)
            except Exception:
                _exp_all = []

            _exp_filtered = _exp_all if selected_part == "全パート" else [r for r in _exp_all if r.get("part") == selected_part]
            _exp_diff = [r for r in _exp_filtered if last_export_ts > 0 and (r.get("createdAt") or 0) > last_export_ts]

            ec1, ec2 = st.columns(2)
            with ec1:
                if _exp_filtered:
                    _json_full = json.dumps(_exp_filtered, ensure_ascii=False, indent=None)
                    _mb = len(_json_full) / 1024 / 1024
                    st.download_button(
                        f"📤 全問 ({len(_exp_filtered)}問 / {_mb:.0f}MB)",
                        _json_full,
                        f"toeic-stock-{selected_part}-full-{datetime.now():%Y%m%d-%H%M}.json",
                        "application/json", key="exp_full")
                    del _json_full
            with ec2:
                if _exp_diff:
                    _json_diff = json.dumps(_exp_diff, ensure_ascii=False, indent=None)
                    _mb_d = len(_json_diff) / 1024 / 1024
                    from datetime import datetime as _dt
                    last_dt = _dt.fromtimestamp(last_export_ts/1000).strftime("%m/%d %H:%M")
                    st.download_button(
                        f"🆕 差分 ({len(_exp_diff)}問 / {_mb_d:.0f}MB)",
                        _json_diff,
                        f"toeic-stock-{selected_part}-diff-{datetime.now():%Y%m%d-%H%M}.json",
                        "application/json", key="exp_diff", help=f"前回: {last_dt}")
                    del _json_diff
                elif last_export_ts > 0:
                    st.button("🆕 差分なし", disabled=True, key="exp_diff_empty")
                else:
                    st.caption("初回は全問で出力")
            del _exp_all, _exp_filtered, _exp_diff

            if st.button("⏱ エクスポート時刻を記録", key="mark_export", help="次回の差分基準を更新"):
                with open(LAST_EXPORT_FILE, "w") as f:
                    f.write(str(int(time.time() * 1000)))
                st.success("✅ 記録しました")
                st.rerun()

            # Delete operations
            st.divider()
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("🗑️ パート削除", key="do_delete"):
                    if selected_part == "全パート":
                        st.session_state.results = []
                        _audio_store.clear()
                    else:
                        st.session_state.results = [r for r in st.session_state.results if r.get("part") != selected_part]
                    save_results(RESULTS_FILE, st.session_state.results)
                    st.session_state.pop("_prac_cache_key", None)
                    st.session_state.prac_idx = 0
                    st.session_state.prac_answered = {}
                    st.rerun()
            with dc2:
                selected_level = st.selectbox("レベル", ["全レベル","beginner","intermediate","advanced"], key="stock_level_filter", label_visibility="collapsed")
            if st.button("🗑️ 選択レベルのみ削除", key="do_delete_level"):
                if selected_level == "全レベル":
                    st.warning("レベルを選択してください")
                else:
                    before = len(st.session_state.results)
                    def matches(r):
                        if selected_part != "全パート" and r.get("part") != selected_part:
                            return False
                        return r.get("level") == selected_level
                    st.session_state.results = [r for r in st.session_state.results if not matches(r)]
                    deleted = before - len(st.session_state.results)
                    if deleted > 0:
                        save_results(RESULTS_FILE, st.session_state.results)
                        st.session_state.pop("_prac_cache_key", None)
                        st.success(f"✅ {deleted}問を削除 ({selected_part} / {selected_level})")
                        st.rerun()
                    else:
                        st.info("該当する問題がありません")



    # ── タイプ修復 ──
    results_count = len(st.session_state.results)
    if results_count > 0:
        no_type = [r for r in st.session_state.results if not r.get("qSet",{}).get("_questionType")]
        if no_type:
            st.divider()
            st.markdown(f"**🏷️ タイプ修復** ({len(no_type)}/{results_count}問にタイプ未設定)")
            repair_mode = st.radio("修復方法", ["📏 ルールベース (即座)", "🤖 AI判定 (API)"], key="repair_mode", horizontal=True, label_visibility="collapsed")
            if st.button("🏷️ タイプ修復を実行", key="type_repair_btn"):
                try:
                    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                        full_data = json.load(f)
                except Exception:
                    full_data = []
                stat = st.empty()
                prog = st.progress(0)
                repaired = 0
                total_to_fix = len([r for r in full_data if not r.get("qSet",{}).get("_questionType")])
                fixed_idx = 0
                import re as _re

                def infer_type_rule(item):
                    qs = item.get("qSet", {})
                    part = item.get("part", "")
                    if qs.get("_questionType") and qs["_questionType"] != "unknown":
                        return qs["_questionType"]
                    if part == "part2":
                        for q in qs.get("questions", []):
                            m = _re.search(r'出題[タイプ]*[:：]\s*([a-z0-9_]+)', q.get("explanation_ja",""))
                            if m: return m.group(1)
                    if part == "part4" and qs.get("talk_type"): return qs["talk_type"]
                    if part in ("part6","part7","part7s","part7d","part7t"):
                        if qs.get("doc_type"): return qs["doc_type"]
                        if qs.get("doc_type_1"): return qs["doc_type_1"]
                    if part == "part5" and qs.get("questions"):
                        ch = [_re.sub(r'^\([A-Da-d]\)\s*','',c).strip().lower() for c in qs["questions"][0].get("choices",[])]
                        preps = {"in","on","at","by","for","to","with","from","of","about","during","since","until","between","among","through","within","toward"}
                        if all(c in preps for c in ch if c): return "preposition_basic"
                        roots = [_re.sub(r'[^a-z]','',c)[:4] for c in ch]
                        if roots and roots[0] and len(roots[0])>=4 and sum(1 for r in roots if r==roots[0])>=3: return "word_form"
                        return "vocab_context"
                    if part in ("part3","part3_3p") and qs.get("conversation"):
                        conv = qs["conversation"].lower()
                        for pattern, ttype in [
                            (r"printer|copier|computer|server|network","office_equipment"),
                            (r"schedule|reschedule|postpone|cancel the meeting","schedule_change"),
                            (r"project|deadline|milestone|progress","project_discussion"),
                            (r"hotel|check.in|check.out|room","hotel_checkin"),
                            (r"restaurant|reserv|table for|menu","restaurant_order"),
                            (r"flight|airport|boarding|luggage","airport_travel"),
                            (r"repair|fix|broken|maintenance","repair_maintenance"),
                            (r"interview|candidate|hiring|resume","hiring_interview"),
                            (r"train|workshop|certification|seminar","training_workshop"),
                            (r"client|contract|proposal|negotiate","client_negotiation"),
                            (r"complaint|refund|dissatisfied|apologize","complaint_resolution"),
                            (r"new employee|orientation|first day","new_employee"),
                            (r"transfer|promotion|department|position","promotion_transfer"),
                            (r"event|conference|venue|catering","event_planning"),
                            (r"market|campaign|advertis|social media","marketing_campaign"),
                            (r"invoice|payment|billing|reimburse","bank_finance"),
                        ]:
                            if _re.search(pattern, conv): return ttype
                        return "office_general"
                    if part == "part1" and qs.get("scene"):
                        scene = qs["scene"].lower()
                        for pattern, ttype in [
                            (r"desk|typing|computer|monitor","office_desk"),(r"meeting|conference","meeting_conference"),
                            (r"restaurant|cafe|dining","restaurant_cafe"),(r"construction|scaffold","construction_site"),
                            (r"warehouse|factory|forklift","warehouse_factory"),(r"park|bench|garden","park_bench"),
                            (r"store|shop|retail","retail_shopping"),(r"train|station|platform","train_station"),
                        ]:
                            if _re.search(pattern, scene): return ttype
                    return None

                if "ルールベース" in repair_mode:
                    for item in full_data:
                        qs = item.get("qSet", {})
                        if qs.get("_questionType") and qs["_questionType"] != "unknown": continue
                        inferred = infer_type_rule(item)
                        if inferred:
                            qs["_questionType"] = inferred
                            repaired += 1
                        fixed_idx += 1
                        if fixed_idx % 50 == 0:
                            stat.info(f"⏳ {fixed_idx}/{total_to_fix}...")
                            prog.progress(fixed_idx / max(total_to_fix, 1))
                    # Save results
                    prog.progress(1.0)
                    for i, item in enumerate(full_data):
                        if i < len(st.session_state.results):
                            qt = item.get("qSet",{}).get("_questionType")
                            if qt:
                                st.session_state.results[i].setdefault("qSet",{})["_questionType"] = qt
                    save_results(RESULTS_FILE, st.session_state.results)
                    stat.success(f"✅ タイプ修復完了: {repaired}問")
                    st.session_state.pop("_export_data", None)
                    st.session_state.pop("_html_export", None)
                else:
                    api_key = st.session_state.get("gemini_key", "")
                    if not api_key:
                        st.error("Gemini APIキーを設定してください")
                        prog.empty()
                    else:
                        # Pass 1: rules for reliable parts only (Part 2/4/6/7 have exact type in data)
                        # Part 1/3/5 are too rough with rules → send to API
                        API_PARTS = {"part1","part3","part3_3p","part5"}
                        for item in full_data:
                            qs = item.get("qSet", {})
                            if qs.get("_questionType") and qs["_questionType"] != "unknown": continue
                            if item.get("part","") in API_PARTS: continue  # Skip — API will handle
                            inferred = infer_type_rule(item)
                            if inferred:
                                qs["_questionType"] = inferred
                                repaired += 1
                        stat.info(f"ルールベース: {repaired}問（Part 2/4/6/7）。Part 1/3/5をAI判定中...")
                        # Pass 2: API for Part 1/3/5 (overwrite rough rule types) + any remaining
                        RULE_TYPES = {"office_general","office_desk","meeting_conference","restaurant_cafe",
                            "construction_site","warehouse_factory","park_bench","retail_shopping","train_station",
                            "airport_terminal","library_bookstore","word_form","preposition_basic","vocab_context",
                            "office_equipment","schedule_change","project_discussion","hotel_checkin",
                            "restaurant_order","airport_travel","repair_maintenance","hiring_interview",
                            "training_workshop","client_negotiation","complaint_resolution","new_employee",
                            "promotion_transfer","event_planning","marketing_campaign","bank_finance"}
                        remaining = [(i, r) for i, r in enumerate(full_data)
                            if not r.get("qSet",{}).get("_questionType")
                            or (r.get("part","") in API_PARTS and r.get("qSet",{}).get("_questionType","") in RULE_TYPES)]
                        BATCH = 10
                        for b in range(0, len(remaining), BATCH):
                            batch = remaining[b:b+BATCH]
                            summaries = []
                            for idx, (fi, item) in enumerate(batch):
                                qs = item.get("qSet", {})
                                part = item.get("part", "?")
                                text = (qs.get("spoken") or qs.get("conversation") or qs.get("talk") or qs.get("sentence") or qs.get("text") or "")[:150]
                                q_text = (qs.get("questions",[{}])[0].get("question",""))[:80] if qs.get("questions") else ""
                                summaries.append(f"{idx+1}. part={part}, content: {text}, q: {q_text}")
                            parts_in_batch = set(item.get("part","") for _,item in batch)
                            type_hints = {}
                            for p in parts_in_batch:
                                pk = p if p in TYPES else ("part7s" if p.startswith("part7") else p)
                                if pk in TYPES:
                                    type_hints[p] = [t["type"] for t in TYPES[pk]][:20]
                            prompt = f"Classify each TOEIC question into its type. Types per part:\n{json.dumps(type_hints)}\n\nQuestions:\n" + "\n".join(summaries) + "\n\nRespond ONLY with a JSON array of type strings. Example: [\"schedule_change\",\"word_form\"]"
                            try:
                                import urllib.request
                                req = urllib.request.Request(
                                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
                                    data=json.dumps({"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.1}}).encode(),
                                    headers={"Content-Type":"application/json"}, method="POST")
                                with urllib.request.urlopen(req, timeout=30) as resp:
                                    rj = json.loads(resp.read())
                                text_resp = rj["candidates"][0]["content"]["parts"][0]["text"].strip().strip("`").strip()
                                if text_resp.startswith("json"): text_resp = text_resp[4:].strip()
                                types_list = json.loads(text_resp)
                                for idx, (fi, item) in enumerate(batch):
                                    if idx < len(types_list) and types_list[idx]:
                                        item["qSet"]["_questionType"] = types_list[idx]
                                        repaired += 1
                            except Exception as e:
                                print(f"[TYPE REPAIR] API error: {e}", flush=True)
                            fixed_idx += len(batch)
                            stat.info(f"⏳ AI判定: {fixed_idx}/{len(remaining)}...")
                            prog.progress(min(1.0, fixed_idx / max(len(remaining), 1)))
                            time.sleep(1.5)
                        # Save results
                        prog.progress(1.0)
                        for i, item in enumerate(full_data):
                            if i < len(st.session_state.results):
                                qt = item.get("qSet",{}).get("_questionType")
                                if qt:
                                    st.session_state.results[i].setdefault("qSet",{})["_questionType"] = qt
                        save_results(RESULTS_FILE, st.session_state.results)
                        stat.success(f"✅ タイプ修復完了: {repaired}問")
                        st.session_state.pop("_export_data", None)
                        st.session_state.pop("_html_export", None)

    # ── 難易度一括判定 ──
    with st.expander("🎯 難易度一括判定（ストックJSON）"):
        st.caption("エクスポートしたストックJSONに難易度スコア(200-990)を付与します")
        diff_file = st.file_uploader("ストックJSONをアップロード", type=["json"], key="diff_rating_file")
        if diff_file:
            import json as _json
            diff_data = _json.loads(diff_file.read().decode("utf-8"))
            if isinstance(diff_data, list):
                total_d = len(diff_data)
                rated_d = sum(1 for it in diff_data if it.get("difficulty"))
                unrated_d = total_d - rated_d
                st.info(f"全{total_d}問 | 判定済み: {rated_d} | 未判定: {unrated_d}")
                
                rate_mode = st.radio("判定モード", ["未判定のみ", "全再判定"], key="diff_rate_mode", horizontal=True)
                
                if st.button(f"🎯 {'未判定'+str(unrated_d)+'問' if rate_mode=='未判定のみ' else '全'+str(total_d)+'問'}を判定", key="diff_rate_btn"):
                    target_d = [it for it in diff_data if not it.get("difficulty")] if rate_mode == "未判定のみ" else diff_data
                    if not target_d:
                        st.success("判定対象がありません")
                    else:
                        gemini_key = st.session_state.get("api_key", "")
                        if not gemini_key:
                            st.error("Gemini APIキーを設定してください")
                        else:
                            prog_d = st.progress(0)
                            stat_d = st.status(f"🎯 {len(target_d)}問を判定中...", expanded=True)
                            DBATCH = 15
                            done_d, errors_d = 0, 0
                            
                            for di in range(0, len(target_d), DBATCH):
                                batch_d = target_d[di:di+DBATCH]
                                texts_d = []
                                for bi, it in enumerate(batch_d):
                                    qs = it.get("qSet", {})
                                    q = qs.get("questions", [{}])[0]
                                    text = q.get("question","") or qs.get("sentence","") or qs.get("spoken","") or ""
                                    if qs.get("conversation"): text += " " + qs["conversation"][:150]
                                    if qs.get("talk"): text += " " + qs["talk"][:150]
                                    if qs.get("text"): text += " " + qs["text"][:150]
                                    choices = " / ".join(q.get("choices",[])) if q.get("choices") else ""
                                    texts_d.append(f"{bi+1}. [{it.get('part','?')}/{it.get('level','unknown')}] {text[:200]} | {choices[:100]}")
                                
                                prompt_d = "You are a TOEIC scoring expert. Rate each question's difficulty (200-990).\nEach question is labeled [partN/level] where level is the INTENDED generation difficulty.\n\nCRITERIA: VOCAB(basic→400, business→650, advanced→850) + GRAMMAR(simple→400, clause→600, subjunctive→850) + INFERENCE(explicit→400, implied→650, cross-ref→800) + DISTRACTORS(obvious→400, plausible→650, tricky→850).\n\nGENERATION LEVEL CALIBRATION:\n- /beginner: range 300-500\n- /intermediate: range 450-700\n- /advanced: range 650-950 (MUST be 650+)\nAn advanced Part 2 with indirect answers = 600-700. An advanced Part 3 with implied meaning = 700-850.\n\nCRITICAL: Do NOT rate an /advanced question below 600. Respect the generation level.\n\nReturn JSON array: [score1, score2, ...]\n\nQuestions:\n" + "\n".join(texts_d)
                                
                                try:
                                    resp_d = generate_text(prompt_d, "gemini", "gemini-2.5-flash", 
                                                        "https://generativelanguage.googleapis.com/v1beta",
                                                        gemini_key)
                                    import re as _re
                                    m_d = _re.search(r'\[[\d\s,.\n]+\]', resp_d)
                                    if m_d:
                                        scores_d = _json.loads(m_d.group())
                                        for j in range(min(len(batch_d), len(scores_d))):
                                            batch_d[j]["difficulty"] = max(200, min(990, round(float(scores_d[j]))))
                                            done_d += 1
                                    else:
                                        nums_d = _re.findall(r'\b\d{3}\b', resp_d)
                                        if len(nums_d) >= len(batch_d) * 0.5:
                                            for j in range(min(len(batch_d), len(nums_d))):
                                                batch_d[j]["difficulty"] = max(200, min(990, int(nums_d[j])))
                                                done_d += 1
                                        else:
                                            errors_d += 1
                                            stat_d.write(f"⚠️ Batch {di//DBATCH+1}: parse failed")
                                except Exception as e:
                                    errors_d += 1
                                    stat_d.write(f"⚠️ Batch {di//DBATCH+1}: {str(e)[:80]}")
                                    time.sleep(3)
                                
                                prog_d.progress(min(1.0, (di+DBATCH) / len(target_d)))
                                stat_d.update(label=f"🎯 {done_d}/{len(target_d)}問 ({errors_d}err)")
                                time.sleep(0.5)
                            
                            stat_d.update(label=f"✅ {done_d}問判定完了 ({errors_d}err)", state="complete")
                            rated_after_d = sum(1 for it in diff_data if it.get("difficulty"))
                            st.success(f"判定完了: {rated_after_d}/{total_d}問に難易度スコア付与済み")
                            st.download_button(
                                f"📥 難易度付きJSONをダウンロード ({rated_after_d}問)",
                                data=_json.dumps(diff_data, ensure_ascii=False),
                                file_name=f"toeic-stock-rated-{datetime.now().strftime('%Y%m%d-%H%M')}.json",
                                mime="application/json"
                            )
                            st.info("💡 HTMLアプリにインポート → 既存問題の難易度が更新されます")

    st.divider()



# ══════════════════════════════════════
# TAB 1: Generate
# ══════════════════════════════════════
with tab_gen:
    sel = MODEL_OPTIONS.get(st.session_state.model_key,{})
    if st.session_state.model_key.startswith("auto"):
        st.caption("Model: 🔄 Auto (per-part recommended)")
    elif sel:
        st.caption(f"Model: {'🖥️' if sel.get('engine')=='ollama' else '☁️'} {sel.get('model','?')}")

    # JSON Import (以前エクスポートしたファイルを読み込み)
    with st.expander("📥 Import previous JSON", expanded=False):
        uploaded = st.file_uploader("以前エクスポートしたJSONファイル", type=["json"], key="import_json", label_visibility="collapsed")
        if uploaded is not None:
            try:
                import json as _json
                content = uploaded.read().decode("utf-8")
                data = _json.loads(content)
                if not isinstance(data, list):
                    st.error("無効なファイル形式（配列である必要があります）")
                else:
                    # Dedup by createdAt + strict validation
                    existing_ts = set(r.get("createdAt") for r in st.session_state.results)
                    added, skipped, rejected = 0, 0, 0
                    for item in data:
                        if not item.get("part") or not item.get("qSet"):
                            rejected += 1
                            continue
                        # Reject broken listening items regardless of TTS settings
                        ok, reason = validate_stock_item(
                            item,
                            require_tts=True,  # listening MUST have audio
                            require_image_for_part1=True,  # Part 1 MUST have image
                            require_image_for_graphic=False,  # graphic uses HTML table, no image needed
                            require_vocab_audio=False  # lenient on vocab for import
                        )
                        if not ok:
                            rejected += 1
                            print(f"[IMPORT] Rejected: {reason}", flush=True)
                            continue
                        ts = item.get("createdAt") or int(time.time()*1000)
                        if ts in existing_ts:
                            skipped += 1
                            continue
                        existing_ts.add(ts)
                        _strip_audio(item)  # Move audio to _audio_store
                        st.session_state.results.append(item)
                        added += 1
                    if added > 0:
                        save_results(RESULTS_FILE, st.session_state.results)  # auto-save after import
                    msg = f"✅ {added}問インポート"
                    if skipped: msg += f" ({skipped}問重複スキップ)"
                    if rejected: msg += f" ⚠️ {rejected}問は音声/画像不完全のため除外"
                    st.success(msg)
                    if added > 0:
                        st.caption("📚 Vocabulary / 🧠 Quiz / 🎧 Listening タブで自動的に利用可能です")
            except Exception as e:
                st.error(f"読み込みエラー: {e}")

    # Auto-check Image for Part 1 and Graphic parts
    _cur_part = st.session_state.get("part", "part5")
    if _cur_part in ("part1", "part3_g", "part4_g", "part7_g"):
        st.session_state.enable_image = True

    with st.form("gen_form", border=False):
        c1,c2 = st.columns([3,2])
        with c1:
            PO = {"part1":"Part 1 — Photographs","part2":"Part 2 — Q&R",
                  "part3":"Part 3 — Conversations","part3_3p":"Part 3 — 3-Person Conv","part3_g":"Part 3 — 🖼️ Graphic",
                  "part4":"Part 4 — Talks","part4_g":"Part 4 — 🖼️ Graphic",
                  "part5":"Part 5 — Incomplete Sentences","part6":"Part 6 — Text Completion",
                  "part7":"Part 7 — Reading","part7_g":"Part 7 — 🖼️ Graphic"}
            st.selectbox("Part", list(PO.keys()), format_func=lambda x:PO[x], key="part")
        with c2:
            LO = {"beginner":"🟢 ~600","intermediate":"🟡 600-800","advanced":"🔴 800+"}
            st.selectbox("Level", list(LO.keys()), format_func=lambda x:LO[x], key="level")

        c3,c4,c5 = st.columns([2,1,1])
        with c3: st.number_input("Questions",1,500,key="count")
        with c4: pass
        with c5: st.checkbox("🖼️ Image", key="enable_image")

        gen_submitted = st.form_submit_button("🚀 Generate", type="primary")

    # Graphic mapping: part*_g → base part + graphic types
    GRAPHIC_MAP = {"part3_g":"part3", "part4_g":"part4", "part7_g":"part7s"}
    pk = st.session_state.part
    base_part = GRAPHIC_MAP.get(pk, pk)  # e.g. "part3_g" → "part3"
    is_graphic_mode = pk in GRAPHIC_MAP
    is_listening = base_part in ("part1","part2","part3","part3_3p","part4") or pk in ("part3_g","part4_g")

    if is_graphic_mode:
        gc = len([t for t in TYPES.get(base_part,[]) if t["type"].startswith("graphic_")])
        st.caption(f"📋 {gc} graphic variations")
    else:
        tc = len(TYPES.get("part7s",[])) + len(TYPES.get("part7d",[])) + len(TYPES.get("part7t",[])) if pk=="part7" else len(TYPES.get(pk,[]))
        st.caption(f"📋 {tc} variations")

    if gen_submitted:
        part,level,count = st.session_state.part, st.session_state.level, st.session_state.count
        url,api_key,tts_eng = st.session_state.ollama_url, st.session_state.gemini_key, st.session_state.tts_engine
        is_auto = st.session_state.model_key.startswith("auto")
        def resolve_model(p):
            if is_auto:
                mk = PART_DEFAULT_MODEL.get(p, "gemini-2.5-flash (API balanced)")
                return MODEL_OPTIONS[mk]
            return MODEL_OPTIONS.get(st.session_state.model_key)
        sel = resolve_model(base_part)
        if not sel: st.error("Select model"); st.stop()
        engine,model = sel["engine"],sel["model"]
        do_tts = tts_eng!="off" and is_listening  # TTS auto-on for listening parts
        do_img = api_key and (part == "part1" or is_graphic_mode or st.session_state.enable_image)

        if is_graphic_mode:
            graphic_pool = [(t, base_part) for t in TYPES.get(base_part,[]) if t["type"].startswith("graphic_")]
            random.shuffle(graphic_pool)
            pool_items = graphic_pool
        else:
            pool = TYPES.get(base_part,[])
            if base_part=="part7": pool = TYPES["part7s"]+TYPES["part7d"]+TYPES["part7t"]
            random.shuffle(pool)
            pool_items = [(t, base_part) for t in pool]

        prog = st.progress(0); stat = st.empty(); log = st.container()
        gen,fail = 0,0
        if engine=="ollama": stat.info(f"⏳ Loading {model}..."); ollama_warmup(url,model)
        mode_str = "auto" if is_auto else f"{engine}:{model}"
        print(f"\n[START] {pk}{'(graphic)' if is_graphic_mode else ''} {level} x{count} {mode_str} tts={tts_eng}", flush=True)
        for i in range(count):
            to, actual_part = pool_items[i % len(pool_items)] if pool_items else ({}, part)
            # For graphic mode, resolve model per actual part
            if is_graphic_mode:
                sel = resolve_model(actual_part)
                if sel: engine,model = sel["engine"],sel["model"]
            tt = to.get('type','?')
            print(f"\n{'━'*50}\n[{i+1}/{count}] {actual_part} ({tt})", flush=True)
            stat.info(f"⏳ [{i+1}/{count}] {actual_part} ({tt})..."); prog.progress(i/count)
            try:
                prompt,ap = build_prompt(level,actual_part,to)
                raw = generate_text(prompt,engine,model,url,api_key)
                qs = enforce_choice_count(normalize_set(parse_json(raw),ap))
                if not qs.get("questions"): raise ValueError("No questions")
                # Consistency check BEFORE shuffle
                if not check_answer_consistency(qs, actual_part):
                    print(f"[RETRY] Answer/explanation mismatch detected, regenerating...", flush=True)
                    time.sleep(2)
                    raw2 = generate_text(prompt,engine,model,url,api_key)
                    qs2 = enforce_choice_count(normalize_set(parse_json(raw2),ap))
                    if qs2.get("questions") and check_answer_consistency(qs2, actual_part):
                        qs = qs2
                    else:
                        print(f"[WARN] Retry also inconsistent, keeping original", flush=True)
                # Shuffle AFTER consistency check
                qs = shuffle_answer_positions(qs)
                qs["_questionType"] = to.get("type", "unknown")  # Save type for diagnosis
                real_part = qs.get("part", actual_part)
                item = {"part":real_part,"level":level,"createdAt":int(time.time()*1000)+i,"qSet":qs,"imgUrl":None,"audioOpus":None}
                if do_tts and qs.get("audio"):
                    tts_part = real_part  # Use actual part, not "graphic"
                    try:
                        at = qs["audio"]
                        if tts_eng=="edge" and check_edge_tts():
                            print(f"[TTS] Part={tts_part}, engine=edge{'_conv' if tts_part=='part3' else ''}", flush=True)
                            mp3 = edge_tts_conv(at,qs.get("speakers")) if tts_part in ("part3","part3_3p") else edge_tts_sync(at,random.choice(EDGE_VF+EDGE_VM))
                            o = mp3_to_opus(mp3)
                            if o:
                                item["audioOpus"]=base64.b64encode(o).decode(); item["audioFormat"]="opus"
                                print(f"[TTS] ✅ OK ({len(o)//1024}KB)", flush=True)
                            else:
                                print(f"[TTS] ❌ Opus encode failed", flush=True)
                        elif tts_eng=="azure" and st.session_state.get("azure_speech_key"):
                            az_key = st.session_state.azure_speech_key
                            az_region = st.session_state.get("azure_speech_region","eastus")
                            print(f"[TTS] Part={tts_part}, engine=azure{'_conv' if tts_part=='part3' else ''}", flush=True)
                            mp3 = azure_tts_conv(at,az_key,az_region,qs.get("speakers")) if tts_part in ("part3","part3_3p") else azure_tts(at,az_key,az_region)
                            o = mp3_to_opus(mp3)
                            if o:
                                item["audioOpus"]=base64.b64encode(o).decode(); item["audioFormat"]="opus"
                                print(f"[TTS] ✅ OK ({len(o)//1024}KB)", flush=True)
                            else:
                                print(f"[TTS] ❌ Azure opus encode failed", flush=True)
                        elif tts_eng=="gemini" and api_key:
                            if tts_part in ("part3","part3_3p"):
                                print(f"[TTS] Part={tts_part}, engine=gemini_conv ({len(qs.get('speakers',[]))} speakers)", flush=True)
                                p = gemini_tts_conv(at,api_key,qs.get("speakers")); o = pcm_to_opus(p)
                            else:
                                print(f"[TTS] Part={tts_part}, engine=gemini", flush=True)
                                p = gemini_tts(at,api_key); o = pcm_to_opus(p)
                            if o:
                                item["audioOpus"]=base64.b64encode(o).decode(); item["audioFormat"]="opus"
                                print(f"[TTS] ✅ OK ({len(o)//1024}KB)", flush=True)
                            else:
                                print(f"[TTS] ❌ Opus encode failed", flush=True)
                            time.sleep(6.5)
                        else:
                            print(f"[TTS] ⏭️ Skipped (engine={tts_eng}, key={'set' if api_key else 'none'})", flush=True)
                    except Exception as e: print(f"[TTS] ❌ {e}",flush=True); log.warning(f"⚠️ TTS: {e}")
                elif not do_tts:
                    is_lp = real_part in ("part1","part2","part3","part3_3p","part4")
                    if is_lp: print(f"[TTS] ⏭️ do_tts=False (tts_eng={tts_eng})", flush=True)
                # Image: Part 1 scene OR graphic mode (both REQUIRED)
                if do_img and api_key:
                    if qs.get("scene"):  # Part 1
                        try: item["imgUrl"]=gemini_image(qs["scene"],api_key); time.sleep(5)
                        except Exception as e: log.warning(f"⚠️ Image: {e}")
                    # Graphic questions: image is REQUIRED (maps, floor plans, charts)
                    elif is_graphic_mode and qs.get("graphic"):
                        g = qs["graphic"]
                        desc = f"Clean business graphic for a test: {g.get('title','')}. "
                        if g.get("headers") and g.get("rows"):
                            desc += f"A table with columns: {', '.join(g['headers'])}. "
                            desc += f"Rows: {'; '.join([', '.join(str(c) for c in r) for r in g['rows'][:3]])}. "
                        desc += "Simple, professional style. No decorations. Clear text. White background."
                        try:
                            print(f"[IMG] Generating graphic: {desc[:80]}...", flush=True)
                            item["imgUrl"]=gemini_image(desc, api_key, size="512")
                            time.sleep(5)
                        except Exception as e: log.warning(f"⚠️ Graphic image: {e}")
                # Vocab audio pre-generation (word + example sentence) — Edge preferred
                if qs.get("vocab") and tts_eng != "off":
                    _use_edge_vocab = check_edge_tts()
                    for vi, vw in enumerate(qs["vocab"]):
                        word = vw.get("word","")
                        ex = vw.get("example","")
                        if not word: continue
                        try:
                            if _use_edge_vocab:
                                # Retry once on edge TTS failure
                                def _edge_with_retry(txt):
                                    for att in range(2):
                                        try:
                                            return edge_tts_sync(txt, random.choice(EDGE_VF+EDGE_VM))
                                        except Exception as ex2:
                                            if att == 0: time.sleep(0.5)
                                            else: raise
                                    return None
                                mp3w = _edge_with_retry(word)
                                if mp3w:
                                    oo = mp3_to_opus(mp3w, '12k')
                                    if oo: vw["audio"] = base64.b64encode(oo).decode()
                                if ex:
                                    try:
                                        mp3e = _edge_with_retry(ex)
                                        if mp3e:
                                            eo = mp3_to_opus(mp3e, '12k')
                                            if eo: vw["example_audio"] = base64.b64encode(eo).decode()
                                    except: pass
                            elif tts_eng=="gemini" and api_key:
                                pp = gemini_tts(word, api_key); oo = pcm_to_opus(pp, '12k')
                                if oo: vw["audio"] = base64.b64encode(oo).decode()
                                time.sleep(4)
                                if ex:
                                    ep = gemini_tts(ex, api_key); eo = pcm_to_opus(ep, '12k')
                                    if eo: vw["example_audio"] = base64.b64encode(eo).decode()
                                    time.sleep(4)
                        except Exception as e:
                            print(f"[VOCAB main] {e}", flush=True)
                    print(f"[VOCAB] Audio: {len([v for v in qs['vocab'] if v.get('audio')])} words, {len([v for v in qs['vocab'] if v.get('example_audio')])} examples ({('Edge' if _use_edge_vocab else 'Gemini')})", flush=True)
                # Listen-mode Q&A audio (Edge TTS — Part 2/3/4 only)
                if actual_part in ("part2","part3","part3_3p","part4") and qs.get("questions") and tts_eng != "off" and check_edge_tts():
                    _generate_listen_audio(qs, actual_part)
                # STRICT VALIDATION before saving to stock
                is_listening_q = actual_part in ("part1","part2","part3","part3_3p","part4")
                ok, reason = validate_stock_item(
                    item,
                    require_tts=(is_listening_q and do_tts),  # Listening音声必須
                    require_image_for_part1=(actual_part == "part1" and do_img),
                    require_image_for_graphic=(is_graphic_mode and do_img),
                    require_vocab_audio=False,  # vocab audio can fallback at playback
                    strict_vocab=False
                )
                if not ok:
                    fail += 1
                    print(f"[SKIP] [{i+1}/{count}] validation failed: {reason}", flush=True)
                    log.warning(f"⏭️ [{i+1}/{count}] {tt} - {reason}")
                    continue
                _strip_audio(item)  # Move audio to _audio_store
                st.session_state.results.append(item); gen+=1
                save_results(RESULTS_FILE, st.session_state.results)  # auto-save
                print(f"[OK] [{i+1}/{count}]",flush=True); log.success(f"✅ [{i+1}/{count}] {tt}")
            except Exception as e:
                fail+=1; print(f"[ERR] {e}",flush=True); log.error(f"❌ [{i+1}/{count}] {e}")
                # Retry is disabled for strict mode - listening needs full regeneration with TTS
                # The main loop will continue and try the next question
        prog.progress(1.0); stat.success(f"🎉 {gen}/{count} generated, {fail} failed")

    # Summary
    if st.session_state.results:
        st.divider(); st.subheader(f"📦 {len(st.session_state.results)} questions")
        pc = {}
        for r in st.session_state.results: p=r.get("part","?"); pc[p]=pc.get(p,0)+1
        cols = st.columns(max(len(pc),1))
        for i,(p,c) in enumerate(pc.items()): cols[i].metric(p,c)

    # ── 模試一括生成 ──
    st.divider()
    st.subheader("🎯 模試一括生成")
    st.caption("本番TOEICの問題構成・難易度ミックスで一括生成します。生成済みは Practice タブと別のセッション変数に保存されます。")

    # Show plan preview
    with st.expander("📊 ミックス内訳を確認", expanded=False):
        st.markdown("""
        **本番想定の難易度ミックス** (易30% / 中50% / 難20% を基本に、パート特性で調整):
        - Part 1: 易60% / 中30% / 難10%（写真は本番で取りやすい）
        - Part 2: 易30% / 中50% / 難20%
        - Part 3: 易20% / 中50% / 難30%（聞き取りが鍵）
        - Part 4: 易20% / 中50% / 難30%（最難関リスニング）
        - Part 5: 易30% / 中50% / 難20%
        - Part 6: 易20% / 中50% / 難30%
        - Part 7: 易20% / 中50% / 難30%（長文読解）
        """)
        # Show concrete plan for 100/200
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**ハーフ模試 (100問) の生成計画:**")
            plan_h = build_mock_plan(scale=0.5)
            cnt_h = {}
            for p, lv in plan_h:
                key = f"{p}/{lv}"
                cnt_h[key] = cnt_h.get(key, 0) + 1
            for p in MOCK_FULL_DIST.keys():
                line = f"- **{p.upper()}**: "
                parts = []
                for lv in ["beginner","intermediate","advanced"]:
                    n = cnt_h.get(f"{p}/{lv}", 0)
                    if n > 0: parts.append(f"{lv[:3]}×{n}")
                line += " / ".join(parts) if parts else "なし"
                st.caption(line)
        with cols[1]:
            st.markdown("**フル模試 (200問) の生成計画:**")
            plan_f = build_mock_plan(scale=1.0)
            cnt_f = {}
            for p, lv in plan_f:
                key = f"{p}/{lv}"
                cnt_f[key] = cnt_f.get(key, 0) + 1
            for p in MOCK_FULL_DIST.keys():
                line = f"- **{p.upper()}**: "
                parts = []
                for lv in ["beginner","intermediate","advanced"]:
                    n = cnt_f.get(f"{p}/{lv}", 0)
                    if n > 0: parts.append(f"{lv[:3]}×{n}")
                line += " / ".join(parts) if parts else "なし"
                st.caption(line)

    mc1, mc2 = st.columns(2)
    with mc1:
        gen_half = st.button("⚡ 模試 100問 (ハーフ) 生成", type="secondary")
    with mc2:
        gen_full = st.button("🏆 模試 200問 (フル) 生成", type="primary")

    st.caption("💡 模試生成では以下が**自動設定**されます（上の Generate 設定は無視）:\n"
               "🤖 LLM: Auto (Part別最適) | 🔊 TTS: Gemini優先→Edge | 🖼️ Image: Gemini API (Part1/Graphic)")

    if gen_half or gen_full:
        scale = 0.5 if gen_half else 1.0
        plan = build_mock_plan(scale=scale)
        # init mock_results storage if needed
        if "mock_results" not in st.session_state:
            st.session_state.mock_results = []

        # Each generation gets a unique batchId — old batches are kept as stock
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_label = f"{'フル200問' if scale >= 1.0 else 'ハーフ100問'} ({datetime.now().strftime('%m/%d %H:%M')})"

        url = st.session_state.ollama_url
        api_key = st.session_state.gemini_key

        # ── 模試では設定を強制オーバーライド ──
        # モデル: 必ず autoモード (PART_DEFAULT_MODEL ベース)
        def resolve_model_for(p):
            mk = PART_DEFAULT_MODEL.get(p, "gemini-2.5-flash (API balanced)")
            return MODEL_OPTIONS[mk]
        # TTS: サイドバーの設定を尊重 (Azure > Edge > Gemini > off)
        tts_eng = st.session_state.tts_engine
        if tts_eng == "off" and (api_key or check_edge_tts()):
            tts_eng = "azure" if st.session_state.get("azure_speech_key") else ("edge" if check_edge_tts() else "off")
        # Image: APIキーがあれば必須ON
        force_image = bool(api_key)

        # 設定確認の表示
        config_info = []
        config_info.append(f"🤖 LLM: Auto (Part別最適 / Ollama+Gemini)")
        if tts_eng == "azure": config_info.append(f"🔊 TTS: ☁️ Azure Speech")
        elif tts_eng == "gemini": config_info.append(f"🔊 TTS: 🌟 Gemini")
        elif tts_eng == "edge": config_info.append(f"🔊 TTS: Edge (無料)")
        else: config_info.append(f"🔊 TTS: ⚠️ なし")
        if force_image: config_info.append(f"🖼️ Image: Gemini Image API (Part1/Graphic)")
        else: config_info.append(f"🖼️ Image: ⚠️ なし (APIキー必要)")
        st.success(" | ".join(config_info))

        # 警告
        warnings = []
        if not api_key:
            warnings.append("⚠️ Gemini APIキー未設定: Part 1/Graphic の画像が生成できません")
            if tts_eng == "off":
                warnings.append("⚠️ TTS無効: Listening音声が生成できません")
        if warnings:
            for w in warnings: st.warning(w)
            if not st.button("⚠️ それでも生成を続行する", type="secondary"):
                st.stop()

        st.info(f"📋 計画: {len(plan)}セット生成 → 本番形式で約{int(sum(MOCK_FULL_DIST.values()) * scale)}問\n\n💡 **音声・画像・問題文すべて揃ったもののみカウント**されます。不完全な問題は自動的にリトライします (各セット最大3回)。")
        prog = st.progress(0)
        stat = st.empty()
        log = st.container()
        gen, fail = 0, 0
        total = len(plan)

        # Pre-warm if any ollama model needed
        warmed = set()
        for part_p, lv in plan:
            sel = resolve_model_for(part_p)
            if sel and sel["engine"] == "ollama" and sel["model"] not in warmed:
                stat.info(f"⏳ Warming up {sel['model']}...")
                ollama_warmup(url, sel["model"])
                warmed.add(sel["model"])

        print(f"\n[MOCK START] batch={batch_id} {len(plan)} sets, scale={scale}, tts={tts_eng}, image={force_image}", flush=True)

        # Each batch generates fresh — no counting of existing items
        target_per_part_level = {}
        for part_p, lv in plan:
            k = (part_p, lv)
            target_per_part_level[k] = target_per_part_level.get(k, 0) + 1

        needed = dict(target_per_part_level)  # generate everything (fresh batch)
        remaining_total = sum(needed.values())
        stock_before = len(st.session_state.mock_results)

        MAX_RETRIES_PER_SET = 3  # 各セット最大3回までリトライ
        idx_global = 0

        # Graphic quota tracking (per part)
        graphic_dist = MOCK_GRAPHIC_SETS_FULL if scale >= 1.0 else MOCK_GRAPHIC_SETS_HALF
        graphic_got = {}   # how many graphic sets generated per part
        graphic_target = {}  # how many needed per part

        for (part_p, lv), need_count in needed.items():
            if need_count <= 0: continue
            sel = resolve_model_for(part_p)
            if not sel:
                fail += need_count
                log.error(f"❌ {part_p}/{lv} - Model not found ({need_count} sets)")
                continue
            engine, model = sel["engine"], sel["model"]

            # Calculate graphic target for this part (shared across levels)
            if part_p not in graphic_target:
                total_sets_for_part = sum(v for (p,l),v in needed.items() if p == part_p)
                graphic_target[part_p] = min(graphic_dist.get(part_p, 0), total_sets_for_part)
                graphic_got[part_p] = 0

            got = 0
            attempts = 0
            max_attempts = need_count * MAX_RETRIES_PER_SET
            consecutive_503 = 0
            FALLBACK_THRESHOLD = 3  # Switch to next model after this many consecutive 503s
            # Determine starting position in fallback chain
            fallback_idx = 0
            for fi, fc in enumerate(GEMINI_FALLBACK_CHAIN):
                if fc["engine"] == engine and fc["model"] == model:
                    fallback_idx = fi
                    break
            if engine == "ollama":
                fallback_idx = len(GEMINI_FALLBACK_CHAIN) - 1  # already at local

            # Pre-plan which sets in this batch should be graphic
            # Spread graphic sets evenly instead of front-loading
            g_needed_here = 0
            g_target_part = graphic_target.get(part_p, 0)
            g_done_part = graphic_got.get(part_p, 0)
            g_remaining = g_target_part - g_done_part
            if g_remaining > 0 and need_count > 0:
                # Proportional share: this batch's fair share of remaining graphic quota
                total_remaining_for_part = sum(v for (p,l),v in needed.items() if p == part_p and v > 0)
                if total_remaining_for_part > 0:
                    g_needed_here = max(0, round(g_remaining * need_count / total_remaining_for_part))
                    g_needed_here = min(g_needed_here, g_remaining, need_count)
            # Pre-select which indices in this batch will be graphic
            graphic_indices = set()
            if g_needed_here > 0:
                indices = list(range(need_count))
                random.shuffle(indices)
                graphic_indices = set(indices[:g_needed_here])
                print(f"[MOCK] {part_p}/{lv}: {g_needed_here}/{need_count} sets will be graphic (indices: {sorted(graphic_indices)})", flush=True)

            # Type queues: shuffle all types, cycle through before repeating (max diversity)
            _type_queue_normal = []
            _type_queue_graphic = []

            while got < need_count and attempts < max_attempts:
                attempts += 1
                idx_global += 1

                pool = TYPES.get(part_p, [])
                if not pool:
                    break

                g_pool = [t for t in pool if t.get("type","").startswith("graphic_")]
                n_pool = [t for t in pool if not t.get("type","").startswith("graphic_")]
                force_graphic = got in graphic_indices and g_pool

                if force_graphic:
                    if not _type_queue_graphic:
                        _type_queue_graphic = list(g_pool); random.shuffle(_type_queue_graphic)
                    to = _type_queue_graphic.pop(0)
                else:
                    if not _type_queue_normal:
                        _type_queue_normal = list(n_pool) if n_pool else list(pool); random.shuffle(_type_queue_normal)
                    to = _type_queue_normal.pop(0)

                actual_part = part_p
                tt = to.get("type", "?")

                # TTS for L parts only (Part 1-4)
                do_tts = (tts_eng != "off") and part_p in ("part1","part2","part3","part3_3p","part4")
                # Image: Part 1 (写真) または graphic_系 (図表)
                do_img = force_image and (part_p == "part1" or tt.startswith("graphic_"))
                is_g = tt.startswith("graphic_")

                engine_label = "🖥️" if engine == "ollama" else "☁️"
                stat.info(f"⏳ [{part_p}/{lv}] {got+1}/{need_count}問目 (全体 {stock_before+gen}/{stock_before+remaining_total}) {engine_label} ({tt}) 試行{attempts}回目...")
                prog.progress(min(gen / max(remaining_total,1), 1.0))
                print(f"\n[MOCK {stock_before+gen+1}/{stock_before+remaining_total}] {part_p}/{lv} ({tt}) try#{attempts} using {engine}:{model}", flush=True)

                try:
                    item = generate_one_question(
                        lv, actual_part, to, engine, model, url, api_key,
                        do_tts, do_img, tts_eng, is_g, idx_seed=idx_global
                    )
                    # STRICT VALIDATION: 音声+画像+問題文+vocab音声すべて揃ってないと無効
                    is_listening_q = actual_part in ("part1","part2","part3","part3_3p","part4")
                    ok, reason = validate_stock_item(
                        item,
                        require_tts=(is_listening_q and do_tts),
                        require_image_for_part1=(actual_part == "part1" and do_img),
                        require_image_for_graphic=(is_g and do_img),
                        require_vocab_audio=False,  # soft - can fallback at playback
                        strict_vocab=False
                    )
                    if not ok:
                        fail += 1
                        print(f"[MOCK SKIP] {part_p}/{lv} ({tt}): {reason}", flush=True)
                        log.warning(f"⏭️ {part_p}/{lv} - {reason} (試行{attempts})")
                        continue
                    # All valid - save
                    item["mock"] = True
                    item["batchId"] = batch_id
                    item["batchLabel"] = batch_label
                    st.session_state.mock_results.append(item)
                    save_mock_batch(batch_id, st.session_state.mock_results)  # auto-save per item
                    gen += 1; got += 1
                    consecutive_503 = 0  # Reset on success
                    if is_g:
                        graphic_got[part_p] = graphic_got.get(part_p, 0) + 1
                    has_audio = "🔊" if get_audio(item) else ""
                    has_img = "🖼️" if item.get("imgUrl") else ""
                    has_graphic = "📊" if is_g else ""
                    log.success(f"✅ [{part_p}/{lv}] {got}/{need_count} ({tt}) {has_audio}{has_img}{has_graphic}")
                except Exception as e:
                    fail += 1
                    err_str = str(e)
                    print(f"[MOCK ERR] {err_str}", flush=True)
                    log.error(f"❌ {part_p}/{lv}: {err_str[:80]} (試行{attempts})")
                    # 503 cascade fallback: Pro → Flash → Flash-Lite → gemma3
                    if "503" in err_str:
                        consecutive_503 += 1
                        if consecutive_503 >= FALLBACK_THRESHOLD and fallback_idx < len(GEMINI_FALLBACK_CHAIN) - 1:
                            fallback_idx += 1
                            fb = GEMINI_FALLBACK_CHAIN[fallback_idx]
                            engine, model = fb["engine"], fb["model"]
                            consecutive_503 = 0
                            max_attempts = max(max_attempts, attempts + need_count * MAX_RETRIES_PER_SET)
                            log.warning(f"🔄 {part_p}: 503 ×{FALLBACK_THRESHOLD} → {fb['label']} にフォールバック")
                            print(f"[FALLBACK] {part_p}: → {fb['label']} (chain {fallback_idx}/{len(GEMINI_FALLBACK_CHAIN)-1})", flush=True)
                    else:
                        consecutive_503 = 0

            if got < need_count:
                log.warning(f"⚠️ {part_p}/{lv}: {got}/{need_count}問のみ成功 (最大試行回数に達しました)")

        prog.progress(1.0)
        stock_after = len(st.session_state.mock_results)
        added = stock_after - stock_before

        # ── Gap-fill: validate batch and generate missing parts ──
        MAX_FILL_ROUNDS = 3
        for fill_round in range(MAX_FILL_ROUNDS):
            # Count what we actually have in this batch
            batch_items = [r for r in st.session_state.mock_results if r.get("batchId") == batch_id]
            got_per_part = {}
            for r in batch_items:
                p = r.get("part","")
                qs_r = r.get("qSet", {})
                # Resolve Part 7 sub-type from qSet flags (normalize_set sets part="part7")
                if p == "part7":
                    if qs_r.get("isTriple"): p = "part7t"
                    elif qs_r.get("isDouble"): p = "part7d"
                    else: p = "part7s"
                # Resolve Part 3 3-person (normalize_set may set part="part3")
                if p == "part3" and len(qs_r.get("speakers",[])) >= 3:
                    p = "part3_3p"
                nq = len(qs_r.get("questions",[]))
                got_per_part[p] = got_per_part.get(p, 0) + nq
            # Compare to targets
            target_dist = {p: max(1, round(q * scale)) for p, q in MOCK_FULL_DIST.items()}
            gaps = {}
            for p, target_q in target_dist.items():
                actual_q = got_per_part.get(p, 0)
                if actual_q < target_q:
                    gaps[p] = target_q - actual_q
            if not gaps:
                break  # All parts complete
            gap_info = ", ".join(f"{p}: {q}問不足" for p, q in sorted(gaps.items()))
            stat.info(f"🔄 補充ラウンド {fill_round+1}/{MAX_FILL_ROUNDS}: {gap_info}")
            print(f"\n[MOCK FILL {fill_round+1}] Gaps: {gaps}", flush=True)
            print(f"[MOCK FILL {fill_round+1}] got_per_part: {dict(got_per_part)}", flush=True)

            for gap_part, gap_q in gaps.items():
                gap_sets = max(1, -(-gap_q // QS_PER_SET[gap_part]))
                sel = resolve_model_for(gap_part)
                if not sel: continue
                engine, model = sel["engine"], sel["model"]
                # Pick level: intermediate
                lv = "intermediate"
                actual_part = gap_part
                pool = TYPES.get(gap_part, [])
                n_pool = [t for t in pool if not t.get("type","").startswith("graphic_")]
                if not n_pool: n_pool = pool
                type_q = list(n_pool); random.shuffle(type_q)

                for si in range(gap_sets):
                    if not type_q:
                        type_q = list(n_pool); random.shuffle(type_q)
                    to = type_q.pop(0)
                    tt = to.get("type","?")
                    actual_part = gap_part
                    if gap_part == "part3_3p": actual_part = "part3_3p"
                    elif gap_part in ("part7s","part7d","part7t"): actual_part = gap_part

                    do_tts = tts_eng != "off" and gap_part in ("part1","part2","part3","part3_3p","part4")
                    is_g = tt.startswith("graphic_")
                    do_img = force_image and (gap_part == "part1" or is_g)
                    idx_global += 1
                    stat.info(f"🔄 補充: {gap_part}/{lv} ({tt})")
                    print(f"[MOCK FILL] {gap_part}/{lv} ({tt})", flush=True)
                    try:
                        item = generate_one_question(lv, actual_part, to, engine, model, url, api_key, do_tts, do_img, tts_eng, is_g, idx_seed=idx_global)
                        is_listening_q = actual_part in ("part1","part2","part3","part3_3p","part4")
                        ok, reason = validate_stock_item(item, require_tts=(is_listening_q and do_tts), require_image_for_part1=(actual_part=="part1" and do_img), require_image_for_graphic=(is_g and do_img), require_vocab_audio=False, strict_vocab=False)
                        if not ok:
                            print(f"[MOCK FILL SKIP] {gap_part}: {reason}", flush=True)
                            continue
                        item["mock"] = True; item["batchId"] = batch_id; item["batchLabel"] = batch_label
                        st.session_state.mock_results.append(item)
                        save_mock_batch(batch_id, st.session_state.mock_results)
                        gen += 1
                        log.success(f"✅ 補充 {gap_part}/{lv} ({tt})")
                    except Exception as e:
                        print(f"[MOCK FILL ERR] {gap_part}: {e}", flush=True)

        # Final summary
        stock_after = len(st.session_state.mock_results)
        added = stock_after - stock_before
        # Graphic summary
        g_summary = ", ".join(f"{p}: {graphic_got.get(p,0)}/{graphic_target.get(p,0)}" for p in sorted(graphic_target.keys()) if graphic_target[p] > 0)
        if added == remaining_total:
            stat.success(f"🎉 模試生成完了: {added}セット追加 (合計{stock_after}セット) ✅全問題・音声・画像OK")
        else:
            stat.warning(f"⚠️ 模試生成: {added}/{remaining_total}セット追加 (合計{stock_after}セット), {fail}回のリトライ")
        if g_summary:
            log.info(f"📊 Graphic問題: {g_summary}")
        # Final per-part validation
        batch_items = [r for r in st.session_state.mock_results if r.get("batchId") == batch_id]
        got_per_part = {}
        for r in batch_items:
            p = r.get("part","")
            qs_r = r.get("qSet", {})
            if p == "part7":
                if qs_r.get("isTriple"): p = "part7t"
                elif qs_r.get("isDouble"): p = "part7d"
                else: p = "part7s"
            if p == "part3" and len(qs_r.get("speakers",[])) >= 3:
                p = "part3_3p"
            nq = len(qs_r.get("questions",[]))
            got_per_part[p] = got_per_part.get(p, 0) + nq
        target_dist = {p: max(1, round(q * scale)) for p, q in MOCK_FULL_DIST.items()}
        summary_lines = []
        all_ok = True
        for p in ["part1","part2","part3","part3_3p","part4","part5","part6","part7s","part7d","part7t"]:
            t = target_dist.get(p, 0)
            g = got_per_part.get(p, 0)
            status = "✅" if g >= t else "❌"
            if g < t: all_ok = False
            summary_lines.append(f"{status} {p}: {g}/{t}問")
        total_got = sum(got_per_part.values())
        total_target = sum(target_dist.values())
        log.info(f"📋 パート別充足状況 ({total_got}/{total_target}問):\n" + "\n".join(summary_lines))

    # Mock results summary
    if st.session_state.get("mock_results"):
        mock = st.session_state.mock_results
        total_q = sum(len(r.get('qSet',{}).get('questions',[])) for r in mock)

        # Group by batch
        batches = {}
        for r in mock:
            bid = r.get("batchId", "legacy")
            if bid not in batches:
                batches[bid] = {"label": r.get("batchLabel", "旧データ"), "items": []}
            batches[bid]["items"].append(r)

        st.markdown(f"**📦 模試ストック: {len(batches)}セット / {total_q}問**")

        for bid, batch in batches.items():
            items = batch["items"]
            nq = sum(len(r.get('qSet',{}).get('questions',[])) for r in items)
            parts = {}
            for r in items:
                p = r.get("part","?")
                parts[p] = parts.get(p,0) + len(r.get('qSet',{}).get('questions',[]))

            with st.expander(f"📋 {batch['label']} — {nq}問 ({len(items)}セット)", expanded=(len(batches)==1)):
                part_str = " / ".join(f"{p}: {c}問" for p,c in sorted(parts.items()))
                st.caption(part_str)
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    # Restore audio before export
                    full_items = []
                    for r in items:
                        full = json.loads(json.dumps(r, ensure_ascii=False))
                        _restore_audio(full)
                        full.pop("_hasAudio", None)
                        full.pop("_hasImage", None)
                        full_items.append(full)
                    batch_json = json.dumps(full_items, ensure_ascii=False, indent=None)
                    st.download_button(
                        "📥 エクスポート",
                        data=batch_json,
                        file_name=f"toeic-mock-{bid}.json",
                        mime="application/json",
                        key=f"exp_{bid}"
                    )
                    del full_items, batch_json
                with bc2:
                    if st.button(f"🗑️ この模試を削除", key=f"del_{bid}"):
                        st.session_state.mock_results = [r for r in mock if r.get("batchId","legacy") != bid]
                        delete_mock_batch(bid)
                        st.rerun()

        # Global actions
        st.divider()
        gc1, gc2 = st.columns(2)
        with gc1:
            # Restore audio for all mock items before export
            all_full = []
            for r in mock:
                full = json.loads(json.dumps(r, ensure_ascii=False))
                _restore_audio(full)
                full.pop("_hasAudio", None)
                full.pop("_hasImage", None)
                all_full.append(full)
            all_json = json.dumps(all_full, ensure_ascii=False, indent=None)
            st.download_button(
                "📥 全模試エクスポート",
                data=all_json,
                file_name=f"toeic-mock-all-{datetime.now().strftime('%Y%m%d-%H%M')}.json",
                mime="application/json"
            )
            del all_full, all_json
        with gc2:
            if st.button("🗑️ 全模試ストックをクリア"):
                st.session_state.mock_results = []
                clear_all_mock_batches()
                st.rerun()


# ══════════════════════════════════════
# TAB 2: Practice
# ══════════════════════════════════════
with tab_practice:
    @_fragment
    def _practice_frag():
        results = st.session_state.results
        if not results:
            st.info("まだ問題がありません。Generate タブで問題を作成してください。")
        else:
            # Cached part list (avoid iterating 1000+ items every click)
            if "_prac_parts" not in st.session_state or st.session_state.get("_prac_len") != len(results):
                st.session_state._prac_parts = sorted(set(r.get("part","?") for r in results))
                st.session_state._prac_len = len(results)
            
            fc1, fc2 = st.columns([1,1])
            with fc1:
                filt_part = st.selectbox("Filter Part", ["All"] + st.session_state._prac_parts, key="prac_filt_part")
            with fc2:
                st.metric("Total", len(results))

            # Cached filtered indices (not full items — just indices)
            cache_key = f"_pf_{filt_part}_{len(results)}"
            if st.session_state.get("_prac_cache_key") != cache_key:
                if filt_part == "All":
                    st.session_state._prac_indices = list(range(len(results)))
                else:
                    st.session_state._prac_indices = [i for i, r in enumerate(results) if r.get("part") == filt_part]
                st.session_state._prac_cache_key = cache_key
            
            indices = st.session_state._prac_indices
            if not indices:
                st.warning("フィルタに一致する問題がありません")
            else:
                if "prac_idx" not in st.session_state: st.session_state.prac_idx = 0
                if "prac_answered" not in st.session_state: st.session_state.prac_answered = {}

                pidx = st.session_state.prac_idx
                if pidx >= len(indices): pidx = 0; st.session_state.prac_idx = 0

                # Navigation
                nav1, nav2, nav3 = st.columns([1,3,1])
                with nav1:
                    if st.button("◀", disabled=pidx<=0, key="prac_prev"):
                        st.session_state.prac_idx = pidx - 1
                        st.session_state.prac_answered = {}
                        st.rerun()
                with nav2:
                    st.markdown(f"<div style='text-align:center;font-size:18px;padding:4px'><b>{pidx+1}</b> / {len(indices)}</div>", unsafe_allow_html=True)
                with nav3:
                    if st.button("▶", disabled=pidx>=len(indices)-1, key="prac_next"):
                        st.session_state.prac_idx = pidx + 1
                        st.session_state.prac_answered = {}
                        st.rerun()

                # Load current item only
                real_idx = indices[pidx]
                item = results[real_idx]
                qs = item.get("qSet", {})
                part = item.get("part","?")
                level = item.get("level","?")
                lv_colors = {"beginner":"🟢","intermediate":"🟡","advanced":"🔴"}
                st.caption(f"{part.upper()} | {lv_colors.get(level,'')} {level}")

                # Audio — use get_audio() (lightweight: audio stored outside session_state)
                _opus = get_audio(item)
                if _opus:
                    try:
                        raw = base64.b64decode(_opus)
                        st.audio(raw, format="audio/webm")
                    except: pass
                elif part in ("part1","part2","part3","part4"):
                    st.warning("⚠️ 音声なし")

                # Image (Part 1)
                if item.get("imgUrl"):
                    st.image(item["imgUrl"])

                # Content display (minimal — no heavy rendering)
                if part == "part1" and qs.get("scene"):
                    st.caption(f"🖼️ Scene: {qs['scene']}")
                elif part in ("part3","part3_3p") and qs.get("conversation"):
                    with st.expander("💬 Conversation", expanded=True):
                        st.text(qs["conversation"])
                    if qs.get("translation_ja"):
                        with st.expander("🇯🇵 和訳"):
                            st.text(qs["translation_ja"])
                elif part == "part4" and qs.get("talk"):
                    with st.expander("🎤 Talk", expanded=True):
                        st.text(qs["talk"])
                    if qs.get("translation_ja"):
                        with st.expander("🇯🇵 和訳"):
                            st.text(qs["translation_ja"])
                elif part == "part6":
                    if qs.get("header"):
                        st.code(qs["header"], language=None)
                    if qs.get("text"):
                        st.markdown(f"<div style='background:#1e293b;color:#e2e8f0;padding:12px;border-radius:8px;font-size:14px;line-height:1.8'>{qs['text']}</div>", unsafe_allow_html=True)
                    if qs.get("translation_ja"):
                        with st.expander("🇯🇵 和訳"):
                            st.text(qs["translation_ja"])
                elif part == "part7":
                    if qs.get("isTriple"):
                        for d in range(1,4):
                            with st.expander(f"📄 Doc {d}: {qs.get(f'doc_type_{d}','')}", expanded=True):
                                if qs.get(f"header_{d}"): st.code(qs[f"header_{d}"], language=None)
                                st.markdown(qs.get(f"text_{d}",""))
                        for d in range(1,4):
                            if qs.get(f"translation_ja_{d}"):
                                with st.expander(f"🇯🇵 和訳 Doc{d}"):
                                    st.text(qs[f"translation_ja_{d}"])
                    elif qs.get("isDouble"):
                        for d in range(1,3):
                            with st.expander(f"📄 Doc {d}: {qs.get(f'doc_type_{d}','')}", expanded=True):
                                if qs.get(f"header_{d}"): st.code(qs[f"header_{d}"], language=None)
                                st.markdown(qs.get(f"text_{d}",""))
                        for d in range(1,3):
                            if qs.get(f"translation_ja_{d}"):
                                with st.expander(f"🇯🇵 和訳 Doc{d}"):
                                    st.text(qs[f"translation_ja_{d}"])
                    else:
                        if qs.get("header"): st.code(qs["header"], language=None)
                        if qs.get("text"):
                            st.markdown(f"<div style='background:#1e293b;color:#e2e8f0;padding:12px;border-radius:8px;font-size:14px;line-height:1.8'>{qs.get('text','')}</div>", unsafe_allow_html=True)
                        if qs.get("translation_ja"):
                            with st.expander("🇯🇵 和訳"):
                                st.text(qs["translation_ja"])

                # Graphic
                if qs.get("graphic"):
                    g = qs["graphic"]
                    with st.expander("📊 Graphic", expanded=True):
                        if g.get("title"): st.caption(g["title"])
                        if g.get("headers") and g.get("rows"):
                            import pandas as pd
                            st.dataframe(pd.DataFrame(g["rows"], columns=g["headers"]), hide_index=True)

                # Questions
                st.divider()
                questions = qs.get("questions", [])
                answered = st.session_state.prac_answered

                for qi, q in enumerate(questions):
                    qkey = f"q_{real_idx}_{qi}"
                    st.markdown(f"**Q{qi+1}.** {q.get('question','')}")
                    choices = q.get("choices", [])
                    correct = q.get("correct", 0)

                    selected = st.radio(
                        f"Answer Q{qi+1}", choices, key=qkey,
                        index=None, label_visibility="collapsed"
                    )

                    if selected is not None and qkey not in answered:
                        answered[qkey] = choices.index(selected) if selected in choices else -1
                        st.session_state.prac_answered = answered

                    if qkey in answered:
                        user_ans = answered[qkey]
                        if user_ans == correct:
                            st.success(f"✅ Correct! {choices[correct]}")
                        else:
                            st.error(f"❌ Wrong. Correct: {choices[correct]}")
                        if q.get("explanation_en"):
                            st.caption(f"💡 {q['explanation_en']}")
                        if q.get("explanation_ja"):
                            st.info(f"📝 {q['explanation_ja']}")

                # Vocabulary (simplified — no inline TTS buttons)
                vocab = qs.get("vocab", [])
                if vocab:
                    with st.expander(f"📚 Key Vocabulary ({len(vocab)})", expanded=False):
                        for vi, v in enumerate(vocab):
                            st.markdown(f"**{v.get('word','')}** ({v.get('pos','')}) — {v.get('ja','')}")
                            if v.get("example"):
                                st.caption(f"  💬 {v['example']}")

    _practice_frag()

# ══════════════════════════════════════
# TAB 3: Vocabulary
# ══════════════════════════════════════
with tab_mock_test:
    @_fragment
    def _mock_frag():
        mock = st.session_state.get("mock_results", [])

        if not mock:
            st.info("📝 模試テストを受けるには、まず **🔧 Generate** タブで模試を生成してください。")
            st.caption("「⚡ 模試 100問 (ハーフ) 生成」 or 「🏆 模試 200問 (フル) 生成」ボタンを使います。")
        else:
            # ── State init ──
            if "mt_active" not in st.session_state:
                st.session_state.mt_active = False
                st.session_state.mt_flat = []
                st.session_state.mt_idx = 0
                st.session_state.mt_answers = {}
                st.session_state.mt_start = 0
                st.session_state.mt_done = False

            # ── Navigation helpers (via callbacks — avoids rerun-before-state-write bugs) ──
            def mt_go(delta):
                st.session_state.mt_idx = max(0, min(st.session_state.mt_idx + delta, len(st.session_state.mt_flat) - 1))

            def mt_finish():
                st.session_state.mt_active = False
                st.session_state.mt_done = True

            def mt_save_answer():
                """Read the radio widget value and store it."""
                idx = st.session_state.mt_idx
                key = f"mt_radio_{idx}"
                if key in st.session_state:
                    val = st.session_state[key]
                    if val is not None:
                        # val is the full option string — map back to index
                        flat = st.session_state.mt_flat
                        if idx < len(flat):
                            item, q_idx = flat[idx]
                            choices = item.get("qSet",{}).get("questions",[])[q_idx].get("choices",[])
                            letters = ["(A)","(B)","(C)","(D)"]
                            options = [f"{letters[i]} {c}" if not c.startswith("(") else c for i,c in enumerate(choices)]
                            if val in options:
                                st.session_state.mt_answers[idx] = options.index(val)

            def mt_next():
                mt_save_answer()
                mt_go(1)

            def mt_prev():
                mt_save_answer()
                mt_go(-1)

            def mt_end():
                mt_save_answer()
                mt_finish()

            # ── Test setup ──
            if not st.session_state.mt_active and not st.session_state.mt_done:
                st.subheader("📝 模試テスト")

                # Group by batch for selection
                batches = {}
                for r in mock:
                    bid = r.get("batchId", "legacy")
                    if bid not in batches:
                        batches[bid] = {"label": r.get("batchLabel", "旧データ"), "items": []}
                    batches[bid]["items"].append(r)

                if len(batches) == 0:
                    st.info("模試データがありません。🔧 Generate タブで生成してください。")
                else:
                    # Batch selector
                    batch_options = {bid: f"{b['label']} ({sum(len(r.get('qSet',{}).get('questions',[])) for r in b['items'])}問)" for bid, b in batches.items()}
                    if "mt_batch" not in st.session_state or st.session_state.mt_batch not in batch_options:
                        st.session_state.mt_batch = list(batch_options.keys())[-1]  # latest batch

                    if len(batch_options) > 1:
                        selected_bid = st.selectbox(
                            "模試を選択",
                            list(batch_options.keys()),
                            format_func=lambda x: batch_options[x],
                            index=list(batch_options.keys()).index(st.session_state.mt_batch),
                            key="mt_batch_select"
                        )
                        st.session_state.mt_batch = selected_bid
                    else:
                        selected_bid = list(batch_options.keys())[0]
                        st.session_state.mt_batch = selected_bid

                    selected_items = batches[selected_bid]["items"]
                    total_q = sum(len(it.get("qSet",{}).get("questions",[])) for it in selected_items)
                    part_counts = {}
                    for it in selected_items:
                        p = it.get("part","?")
                        nq = len(it.get("qSet",{}).get("questions",[]))
                        part_counts[p] = part_counts.get(p,0) + nq

                    st.markdown(f"**{total_q}問**")
                    cols = st.columns(min(len(part_counts),7))
                    for i,(p,c) in enumerate(sorted(part_counts.items())):
                        cols[i % len(cols)].metric(p.upper(), f"{c}問")

                    st.caption("問題は本番同様の順序 (Part 1→2→3→4→5→6→7) で出題されます")

                    if st.button("🚀 テスト開始", type="primary"):
                        PART_ORDER = {"part1":1,"part2":2,"part3":3,"part3_3p":[{"type": "team_meeting", "desc": "3 colleagues discussing project, deadline, or task assignment"}, {"type": "office_move", "desc": "3 people coordinating office relocation or desk arrangement"}, {"type": "event_coordination", "desc": "3 people planning a company event, party, or conference"}, {"type": "client_presentation", "desc": "3 colleagues preparing for or debriefing a client meeting"}, {"type": "hiring_decision", "desc": "3 people discussing job candidates or interview results"}, {"type": "budget_review", "desc": "3 people reviewing department budget or expense approval"}, {"type": "travel_arrangement", "desc": "3 colleagues arranging a group business trip"}, {"type": "training_feedback", "desc": "3 people discussing training session, workshop, or seminar"}, {"type": "lunch_plans", "desc": "3 coworkers deciding where to eat or planning a lunch meeting"}, {"type": "problem_solving", "desc": "3 people troubleshooting a technical, logistics, or customer issue"}],"part4":4,"part5":5,"part6":6,"part7s":7,"part7d":8,"part7t":9,"part7":7}
                        sorted_items = sorted(selected_items, key=lambda x: PART_ORDER.get(x.get("part","part5"),5))
                        flat = []
                        for it in sorted_items:
                            qs_list = it.get("qSet",{}).get("questions",[])
                            for qi in range(len(qs_list)):
                                flat.append((it, qi))
                        # Trim to target (200 for full, 100 for half — rounding can produce +1)
                        target = 200 if total_q > 150 else 100
                        if len(flat) > target:
                            flat = flat[:target]
                        st.session_state.mt_flat = flat
                        st.session_state.mt_idx = 0
                        st.session_state.mt_answers = {}
                        st.session_state.mt_start = time.time()
                        st.session_state.mt_active = True
                        st.session_state.mt_done = False
                        st.rerun()

            # ── Test in progress ──
            elif st.session_state.mt_active:
                flat = st.session_state.mt_flat
                idx = st.session_state.mt_idx
                total = len(flat)

                if idx >= total:
                    mt_finish()
                    st.rerun()
                else:
                    item, q_idx = flat[idx]
                    part = item.get("part","")
                    qs = item.get("qSet",{})
                    questions = qs.get("questions",[])
                    q = questions[q_idx] if q_idx < len(questions) else {}

                    elapsed = time.time() - st.session_state.mt_start
                    elapsed_min = int(elapsed // 60)
                    elapsed_sec = int(elapsed % 60)

                    st.progress(idx / total)

                    is_listening = part in ("part1","part2","part3","part3_3p","part4")
                    section = "🎧 Listening" if is_listening else "📖 Reading"
                    part_label = part.upper()
                    for old, new in [("PART7S","Part 7-Single"),("PART7D","Part 7-Double"),("PART7T","Part 7-Triple")]:
                        part_label = part_label.replace(old, new)

                    hc1, hc2 = st.columns([3,1])
                    with hc1:
                        st.markdown(f"### {section} — {part_label}")
                        st.caption(f"問題 {idx+1} / {total}")
                    with hc2:
                        st.metric("⏱️", f"{elapsed_min}:{elapsed_sec:02d}")

                    # ── Audio ──
                    _mock_opus = get_audio(item)
                    if is_listening and _mock_opus:
                        try:
                            raw = base64.b64decode(_mock_opus)
                            st.audio(raw, format="audio/webm")
                        except: pass
                    elif is_listening:
                        st.warning("⚠️ この問題には音声データがありません")

                    # ── Image ──
                    if item.get("imgUrl"):
                        st.image(item["imgUrl"])

                    # ── Graphic table ──
                    if qs.get("graphic"):
                        g = qs["graphic"]
                        if g.get("title"): st.caption(f"📊 {g['title']}")
                        if g.get("headers") and g.get("rows"):
                            import pandas as pd
                            df = pd.DataFrame(g["rows"], columns=g["headers"])
                            st.dataframe(df, hide_index=True)

                    # ── Reading passage ──
                    if not is_listening:
                        txt = qs.get("text","") or qs.get("passage","") or qs.get("content","")
                        if txt:
                            expanded = (q_idx == 0)
                            with st.expander("📄 読解パッセージ", expanded=expanded):
                                st.markdown(txt)

                    # ── Question + Choices ──
                    st.markdown(f"**Q{idx+1}. {q.get('question','')}**")

                    choices = q.get("choices",[])
                    letters = ["(A)","(B)","(C)","(D)"]
                    options = [f"{letters[i]} {c}" if not c.startswith("(") else c for i,c in enumerate(choices)]

                    prev_ans = st.session_state.mt_answers.get(idx, None)
                    st.radio(
                        "回答を選択:",
                        options,
                        index=prev_ans,
                        key=f"mt_radio_{idx}",
                        label_visibility="collapsed"
                    )

                    # ── Navigation buttons (use on_click callbacks to save before moving) ──
                    nc1, nc2, nc3 = st.columns([1,1,1])
                    with nc1:
                        if idx > 0:
                            st.button("← 前の問題", on_click=mt_prev)
                    with nc2:
                        answered = len(st.session_state.mt_answers)
                        st.caption(f"回答済: {answered}/{total}")
                    with nc3:
                        if idx < total - 1:
                            st.button("次の問題 →", on_click=mt_next, type="primary")
                        else:
                            st.button("✅ テスト終了", on_click=mt_end, type="primary")

                    st.divider()
                    st.button("🛑 テスト中断 → 結果を見る", on_click=mt_end)

            # ── Results ──
            elif st.session_state.mt_done:
                st.subheader("📊 模試テスト結果")
                flat = st.session_state.mt_flat
                answers = st.session_state.mt_answers
                elapsed = time.time() - st.session_state.mt_start if st.session_state.mt_start else 0
                total = len(flat)
                answered = len(answers)

                correct = 0
                part_correct = {}
                part_total = {}
                wrong_items = []

                for i, (item, q_idx) in enumerate(flat):
                    p = item.get("part","?")
                    qs_list = item.get("qSet",{}).get("questions",[])
                    q = qs_list[q_idx] if q_idx < len(qs_list) else {}
                    correct_ans = q.get("correct", 0)

                    key = p if not p.startswith("part7") else "part7"
                    part_total[key] = part_total.get(key, 0) + 1

                    if i in answers:
                        if answers[i] == correct_ans:
                            correct += 1
                            part_correct[key] = part_correct.get(key, 0) + 1
                        else:
                            wrong_items.append((i, item, q_idx, answers[i], correct_ans))

                # TOEIC score estimation (公式問題集5 換算表ベース — HTML版と同一)
                def score_from_raw(raw_pct, is_listening):
                    """Convert raw percentage (0-100) to TOEIC score (5-495) using interpolation table."""
                    raw_pct = max(0, min(100, raw_pct))
                    if is_listening:
                        table = [(0,5),(5,15),(10,30),(15,45),(20,60),(25,80),(30,110),(35,135),(40,170),
                                 (45,210),(50,250),(55,290),(60,330),(65,370),(70,400),(75,425),(80,455),
                                 (85,475),(90,485),(95,495),(100,495)]
                    else:
                        table = [(0,5),(5,15),(10,25),(15,45),(20,60),(25,75),(30,95),(35,120),(40,150),
                                 (45,185),(50,220),(55,255),(60,290),(65,325),(70,355),(75,385),(80,420),
                                 (85,445),(90,470),(95,485),(100,495)]
                    for i in range(len(table) - 1):
                        r1, s1 = table[i]
                        r2, s2 = table[i + 1]
                        if r1 <= raw_pct <= r2:
                            ratio = (raw_pct - r1) / (r2 - r1) if r2 != r1 else 0
                            return round((s1 + (s2 - s1) * ratio) / 5) * 5  # 5点刻み
                    return 5

                l_parts = ["part1","part2","part3","part4"]
                r_parts = ["part5","part6","part7"]
                l_total = sum(part_total.get(p,0) for p in l_parts)
                l_correct = sum(part_correct.get(p,0) for p in l_parts)
                r_total = sum(part_total.get(p,0) for p in r_parts)
                r_correct = sum(part_correct.get(p,0) for p in r_parts)

                l_pct = round(l_correct / max(l_total, 1) * 100)
                r_pct = round(r_correct / max(r_total, 1) * 100)
                l_score = score_from_raw(l_pct, True)
                r_score = score_from_raw(r_pct, False)
                total_score = l_score + r_score

                elapsed_min = int(elapsed // 60)
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🏆 推定スコア", f"{total_score}", f"L{l_score} + R{r_score}")
                sc2.metric("正答率", f"{correct}/{answered}", f"{correct/max(answered,1)*100:.0f}%")
                sc3.metric("⏱️ 所要時間", f"{elapsed_min}分")

                # Part breakdown
                st.markdown("#### パート別正答率")
                part_order = ["part1","part2","part3","part4","part5","part6","part7"]
                part_names = {"part1":"Part 1","part2":"Part 2","part3":"Part 3","part3_3p":"Part 3 — 3-Person Conv","part4":"Part 4","part5":"Part 5","part6":"Part 6","part7":"Part 7"}
                cols = st.columns(7)
                for i, p in enumerate(part_order):
                    pt = part_total.get(p, 0)
                    pc = part_correct.get(p, 0)
                    rate = f"{pc/pt*100:.0f}%" if pt > 0 else "—"
                    cols[i].metric(part_names.get(p,p), f"{pc}/{pt}", rate)

                # Wrong answers review
                if wrong_items:
                    st.markdown("#### ❌ 間違えた問題")
                    for wi, (i, item, q_idx, user_ans, correct_ans) in enumerate(wrong_items[:50]):
                        p = item.get("part","?")
                        qs_list = item.get("qSet",{}).get("questions",[])
                        q = qs_list[q_idx] if q_idx < len(qs_list) else {}
                        choices = q.get("choices",[])
                        letters = ["A","B","C","D"]

                        with st.expander(f"Q{i+1} ({p.upper()}) — {q.get('question','')[:60]}...", expanded=False):
                            st.markdown(f"**問題:** {q.get('question','')}")
                            for ci, ch in enumerate(choices):
                                pfx = "✅ " if ci == correct_ans else ("❌ " if ci == user_ans else "　 ")
                                st.markdown(f"{pfx}({letters[ci]}) {ch}")
                            expl = q.get("explanation_ja","")
                            if expl: st.info(f"💡 {expl}")
                            expl_en = q.get("explanation_en","")
                            if expl_en: st.caption(f"🇬🇧 {expl_en}")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔄 もう一度受験"):
                        st.session_state.mt_done = False
                        st.session_state.mt_active = False
                        st.rerun()
                with c2:
                    if st.button("🏠 テスト設定に戻る"):
                        st.session_state.mt_done = False
                        st.session_state.mt_active = False
                        st.session_state.mt_flat = []
                        st.session_state.mt_answers = {}
                        st.rerun()
    _mock_frag()
