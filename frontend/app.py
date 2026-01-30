"""
 ML Automation - Professional SaaS Frontend
Enterprise-grade Machine Learning Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from io import StringIO

# Page Configuration
st.set_page_config(
    page_title="ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Euron Theme CSS
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables - Euron Design System */
    :root {
        --primary-blue: #0A66C2;
        --primary-hover: #004182;
        --background: #F3F6F8;
        --surface: #FFFFFF;
        --border: #E5E7EB;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --success: #057642;
        --warning: #B45309;
        --error: #B91C1C;
    }
    
    /* Global styles */
    .stApp {
        background-color: #F3F6F8;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
    }
    
    /* ALL TEXT SHOULD BE DARK BY DEFAULT */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #111827 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    
    /* Card styling */
    .euron-card {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #111827 !important;
    }
    
    .euron-card-header {
        font-size: 18px;
        font-weight: 600;
        color: #111827 !important;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E5E7EB;
    }
    
    /* Header styling */
    .euron-header {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    
    .euron-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.5rem;
    }
    
    .euron-logo-icon {
        width: 40px;
        height: 40px;
        background: #0A66C2 !important;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        font-weight: 700;
        font-size: 20px;
    }
    
    .euron-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827 !important;
        margin: 0;
        line-height: 1.2;
    }
    
    .euron-subtitle {
        font-size: 14px;
        color: #6B7280 !important;
        margin: 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: left;
    }
    
    .metric-label {
        font-size: 12px;
        color: #6B7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827 !important;
    }
    
    .metric-value-blue {
        color: #0A66C2 !important;
    }
    
    /* Best model highlight */
    .best-model-card {
        background: #FFFFFF !important;
        border: 2px solid #0A66C2;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .best-model-badge {
        display: inline-block;
        background: #0A66C2 !important;
        color: white !important;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .best-model-name {
        font-size: 20px;
        font-weight: 600;
        color: #111827 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: #0A66C2 !important;
        color: white !important;
        border: none;
        border-radius: 999px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        transition: background-color 150ms ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #004182 !important;
        color: white !important;
    }
    
    .stButton > button:active {
        background: #004182 !important;
        color: white !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #0A66C2 !important;
        color: white !important;
    }
    
    /* Secondary button */
    .secondary-btn > button {
        background: transparent !important;
        color: #0A66C2 !important;
        border: 1px solid #0A66C2 !important;
    }
    
    .secondary-btn > button:hover {
        background: #F3F6F8 !important;
        color: #004182 !important;
        border-color: #004182 !important;
    }
    
    /* Form inputs */
    .stTextInput > div > div > input {
        border: 1px solid #D1D5DB !important;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        color: #111827 !important;
        background: #FFFFFF !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0A66C2 !important;
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.1);
    }
    
    /* Selectbox and Multiselect - CRITICAL FIXES FOR DROPDOWN VISIBILITY */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
        background: #FFFFFF !important;
        color: #111827 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #111827 !important;
    }
    
    /* Dropdown menu/popup styling - FIX VISIBILITY */
    [data-baseweb="popup"],
    [data-baseweb="menu"],
    [role="listbox"],
    [data-baseweb="select"] [role="listbox"],
    .stSelectbox [data-baseweb="popup"],
    .stMultiSelect [data-baseweb="popup"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Dropdown option items */
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] > div,
    [role="option"],
    [data-baseweb="select"] [role="option"],
    .stSelectbox [data-baseweb="menu"] li,
    .stSelectbox [data-baseweb="menu"] > div,
    .stMultiSelect [data-baseweb="menu"] li,
    .stMultiSelect [data-baseweb="menu"] > div {
        background: #FFFFFF !important;
        color: #111827 !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Dropdown option hover state */
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] > div:hover,
    [role="option"]:hover,
    [data-baseweb="select"] [role="option"]:hover,
    .stSelectbox [data-baseweb="menu"] li:hover,
    .stSelectbox [data-baseweb="menu"] > div:hover,
    .stMultiSelect [data-baseweb="menu"] li:hover,
    .stMultiSelect [data-baseweb="menu"] > div:hover {
        background: #F3F6F8 !important;
        color: #111827 !important;
    }
    
    /* Dropdown option selected state */
    [data-baseweb="menu"] li[aria-selected="true"],
    [role="option"][aria-selected="true"],
    [data-baseweb="select"] [role="option"][aria-selected="true"] {
        background: rgba(10, 102, 194, 0.1) !important;
        color: #0A66C2 !important;
    }
    
    /* Dropdown option text */
    [data-baseweb="menu"] li *,
    [data-baseweb="menu"] > div *,
    [role="option"] *,
    [data-baseweb="select"] [role="option"] *,
    .stSelectbox [data-baseweb="menu"] *,
    .stMultiSelect [data-baseweb="menu"] * {
        color: #111827 !important;
        background: transparent !important;
    }
    
    /* Ensure dropdown text is always visible */
    [data-baseweb="popup"] *,
    [data-baseweb="menu"] *,
    [role="listbox"] * {
        color: #111827 !important;
    }
    
    /* Fix BaseWeb dropdown styles */
    [class*="baseweb"] [data-baseweb="popup"],
    [class*="baseweb"] [data-baseweb="menu"] {
        background: #FFFFFF !important;
    }
    
    [class*="baseweb"] [data-baseweb="popup"] *,
    [class*="baseweb"] [data-baseweb="menu"] * {
        color: #111827 !important;
    }
    
    /* Fix any dark theme overrides */
    [data-baseweb="popup"][style*="background"],
    [data-baseweb="menu"][style*="background"] {
        background: #FFFFFF !important;
    }
    
    /* Multiselect tags/chips */
    [data-baseweb="tag"] {
        background: #0A66C2 !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
    }
    
    [data-baseweb="tag"] * {
        color: #FFFFFF !important;
    }
    
    /* Ensure select input text is visible */
    [data-baseweb="select"] input,
    [data-baseweb="select"] [role="combobox"] {
        background: #FFFFFF !important;
        color: #111827 !important;
    }
    
    [data-baseweb="select"] input::placeholder {
        color: #6B7280 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0A66C2;
    }
    
    [data-testid="stFileUploader"] * {
        color: #111827 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #111827 !important;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        padding: 0.5rem 1rem;
        background: transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0A66C2 !important;
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: #0A66C2 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: transparent !important;
    }
    
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 14px;
        color: #111827 !important;
    }
    
    .stRadio label {
        color: #111827 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #111827 !important;
    }
    
    /* Selectbox label */
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label {
        font-size: 14px;
        font-weight: 500;
        color: #111827 !important;
    }
    
    /* Ensure selectbox value text is visible */
    [data-baseweb="select"] [data-baseweb="select-value"],
    [data-baseweb="select"] [data-baseweb="select-value"] * {
        color: #111827 !important;
        background: transparent !important;
    }
    
    /* Fix selectbox placeholder */
    [data-baseweb="select"] [placeholder],
    [data-baseweb="select"] ::placeholder {
        color: #6B7280 !important;
    }
    
    /* Slider */
    .stSlider label {
        color: #111827 !important;
    }
    
    .stSlider [data-testid="stTickBar"] {
        background: #E5E7EB !important;
    }
    
    .stSlider [data-testid="stThumbValue"] {
        color: #0A66C2 !important;
        font-weight: 600;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #111827 !important;
    }
    
    .stNumberInput input {
        color: #111827 !important;
        background: #FFFFFF !important;
    }
    
    /* DataFrame styling - CRITICAL FIXES FOR VISIBILITY */
    .stDataFrame {
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #FFFFFF !important;
    }
    
    /* DataFrame header - light background with dark text */
    .stDataFrame thead tr th {
        background: #F3F6F8 !important;
        color: #111827 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E5E7EB !important;
        padding: 0.75rem !important;
    }
    
    .stDataFrame thead tr th * {
        color: #111827 !important;
    }
    
    /* DataFrame body */
    .stDataFrame tbody tr td {
        background: #FFFFFF !important;
        color: #111827 !important;
        border-bottom: 1px solid #E5E7EB !important;
        padding: 0.75rem !important;
    }
    
    .stDataFrame tbody tr td * {
        color: #111827 !important;
    }
    
    .stDataFrame tbody tr:hover td {
        background: #F9FAFB !important;
    }
    
    /* Ensure all table text is visible */
    table, table *, thead, thead *, tbody, tbody *, tr, tr *, td, td *, th, th * {
        color: #111827 !important;
    }
    
    /* Override any dark table themes */
    .stDataFrame table {
        background: #FFFFFF !important;
        color: #111827 !important;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background: rgba(5, 118, 66, 0.1) !important;
        border: 1px solid #057642 !important;
        border-radius: 8px;
    }
    
    .stSuccess * {
        color: #057642 !important;
    }
    
    .stError {
        background: rgba(185, 28, 28, 0.1) !important;
        border: 1px solid #B91C1C !important;
        border-radius: 8px;
    }
    
    .stError * {
        color: #B91C1C !important;
    }
    
    .stWarning {
        background: rgba(180, 83, 9, 0.1) !important;
        border: 1px solid #B45309 !important;
        border-radius: 8px;
    }
    
    .stWarning * {
        color: #B45309 !important;
    }
    
    .stInfo {
        background: rgba(10, 102, 194, 0.1) !important;
        border: 1px solid #0A66C2 !important;
        border-radius: 8px;
    }
    
    .stInfo * {
        color: #0A66C2 !important;
    }
    
    /* Expander - CRITICAL FIXES FOR VISIBILITY */
    .streamlit-expanderHeader {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        color: #111827 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #F9FAFB !important;
        border-color: #0A66C2 !important;
    }
    
    .streamlit-expanderHeader * {
        color: #111827 !important;
    }
    
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span,
    .streamlit-expanderHeader div,
    .streamlit-expanderHeader label {
        color: #111827 !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        color: #111827 !important;
    }
    
    .streamlit-expanderContent * {
        color: #111827 !important;
    }
    
    /* Force all expander text to be dark */
    [data-testid="stExpander"] .streamlit-expanderHeader {
        color: #111827 !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderHeader * {
        color: #111827 !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent {
        color: #111827 !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent * {
        color: #111827 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #111827 !important;
    }
    
    .stMarkdown strong {
        color: #111827 !important;
        font-weight: 600;
    }
    
    .stMarkdown code {
        background: #F3F6F8 !important;
        color: #0A66C2 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Code blocks - FIX VISIBILITY */
    .stCodeBlock {
        background: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stCodeBlock code,
    .stCodeBlock pre,
    .stCodeBlock * {
        color: #F3F6F8 !important;
        background: transparent !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    
    /* Inline code */
    code:not(.stCodeBlock code) {
        background: #F3F6F8 !important;
        color: #0A66C2 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #111827 !important;
        margin-bottom: 1rem;
    }
    
    /* Caption text */
    .caption-text {
        font-size: 12px;
        color: #6B7280 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        color: #111827 !important;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 1.5rem 0;
    }
    
    /* Sidebar navigation */
    .nav-item {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: background-color 150ms ease;
        color: #111827 !important;
        font-size: 14px;
        font-weight: 500;
    }
    
    .nav-item:hover {
        background: #F3F6F8 !important;
    }
    
    .nav-item-active {
        background: rgba(10, 102, 194, 0.1);
        color: #0A66C2;
    }
    
    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    /* Caption text */
    .caption-text {
        font-size: 12px;
        color: #6B7280;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 1.5rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F6F8;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9CA3AF;
    }
    
    /* Hide default streamlit elements styling */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Plotly chart background */
    .js-plotly-plot {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }
    
    /* CATCH-ALL FOR TEXT VISIBILITY - Force all text to be dark */
    .main *:not(button):not(.stButton):not(.stDownloadButton):not(.stSuccess):not(.stError):not(.stWarning):not(.stInfo) {
        color: #111827 !important;
    }
    
    /* Ensure headings are always visible */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Force all paragraph and span text to be dark */
    .main p, .main span, .main div, .main label {
        color: #111827 !important;
    }
    
    /* Exception for muted text */
    .caption-text, .euron-subtitle, .metric-label {
        color: #6B7280 !important;
    }
    
    /* Ensure all list items are visible */
    .main ul, .main ol, .main li {
        color: #111827 !important;
    }
    
    /* Fix any Streamlit internal text */
    [class*="st"], [class*="streamlit"] {
        color: #111827 !important;
    }
    
    /* Override any light text on light backgrounds */
    .element-container, .element-container * {
        color: #111827 !important;
    }
    
    /* AGGRESSIVE FIXES FOR EXPANDER CONTENT */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stExpander"] > div {
        background: #FFFFFF !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderHeader {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: none !important;
        border-bottom: 1px solid #E5E7EB !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderHeader:hover {
        background: #F9FAFB !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderHeader * {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent {
        background: #FFFFFF !important;
        color: #111827 !important;
        padding: 1rem !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent * {
        color: #111827 !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent p,
    [data-testid="stExpander"] .streamlit-expanderContent span,
    [data-testid="stExpander"] .streamlit-expanderContent div,
    [data-testid="stExpander"] .streamlit-expanderContent label,
    [data-testid="stExpander"] .streamlit-expanderContent strong {
        color: #111827 !important;
    }
    
    /* Fix markdown inside expanders */
    [data-testid="stExpander"] .stMarkdown,
    [data-testid="stExpander"] .stMarkdown * {
        color: #111827 !important;
    }
    
    [data-testid="stExpander"] .stMarkdown strong {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stExpander"] .stMarkdown code {
        background: #F3F6F8 !important;
        color: #0A66C2 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* Fix write() output inside expanders */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        color: #111827 !important;
    }
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
        color: #111827 !important;
    }
    
    /* Fix columns inside expanders */
    [data-testid="stExpander"] [data-testid="column"] {
        background: transparent !important;
    }
    
    [data-testid="stExpander"] [data-testid="column"] * {
        color: #111827 !important;
    }
    
    /* Ensure all text in main content area is visible */
    .main .block-container {
        color: #111827 !important;
    }
    
    .main .block-container * {
        color: #111827 !important;
    }
    
    /* Fix any remaining light text */
    .stApp > div > div > div > div > div {
        color: #111827 !important;
    }
    
    /* Force visibility on all text elements */
    body, html, #root, .stApp {
        color: #111827 !important;
    }
    
    /* Specific fix for "Individual Model Details" heading */
    .main h3, .main h4 {
        color: #111827 !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Fix any nested divs with light text */
    div[style*="color"] {
        color: #111827 !important;
    }
    
    /* Override Streamlit's default text colors */
    .stMarkdownContainer, .stMarkdownContainer * {
        color: #111827 !important;
    }
    
    /* Ensure all write() outputs are visible */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: #111827 !important;
    }
    
    /* FINAL CATCH-ALL - Override any remaining light text */
    * {
        color: inherit;
    }
    
    /* Force dark text on all non-interactive elements */
    p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th {
        color: #111827 !important;
    }
    
    /* Exception only for specific muted elements */
    .caption-text, .euron-subtitle, .metric-label, 
    [class*="caption"], [class*="muted"], [class*="secondary"] {
        color: #6B7280 !important;
    }
    
    /* Ensure all Streamlit text elements are visible */
    .stText, .stMarkdown, .stWrite, .stDataFrame,
    .stExpander, .stContainer, .stColumn {
        color: #111827 !important;
    }
    
    .stText *, .stMarkdown *, .stWrite *, .stDataFrame *,
    .stExpander *, .stContainer *, .stColumn * {
        color: #111827 !important;
    }
    
    /* Fix any inline styles that might override */
    [style*="color:"]:not([style*="color: #111827"]):not([style*="color: #0A66C2"]):not([style*="color: #6B7280"]):not([style*="color: #057642"]):not([style*="color: #B45309"]):not([style*="color: #B91C1C"]) {
        color: #111827 !important;
    }
    
    /* ADDITIONAL DROPDOWN FIXES - Target all possible dropdown containers */
    div[data-baseweb="popup"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    
    /* All dropdown items */
    div[data-baseweb="popup"] > div,
    div[data-baseweb="menu"] > div,
    ul[role="listbox"] > li,
    div[role="listbox"] > div,
    div[role="listbox"] > li {
        background: #FFFFFF !important;
        color: #111827 !important;
        padding: 0.75rem 1rem !important;
        cursor: pointer !important;
    }
    
    div[data-baseweb="popup"] > div:hover,
    div[data-baseweb="menu"] > div:hover,
    ul[role="listbox"] > li:hover,
    div[role="listbox"] > div:hover,
    div[role="listbox"] > li:hover {
        background: #F3F6F8 !important;
        color: #111827 !important;
    }
    
    /* Selected dropdown item */
    div[data-baseweb="popup"] > div[aria-selected="true"],
    div[data-baseweb="menu"] > div[aria-selected="true"],
    ul[role="listbox"] > li[aria-selected="true"],
    div[role="listbox"] > div[aria-selected="true"],
    div[role="listbox"] > li[aria-selected="true"] {
        background: rgba(10, 102, 194, 0.1) !important;
        color: #0A66C2 !important;
    }
    
    /* Force all text in dropdowns to be dark */
    div[data-baseweb="popup"] *,
    div[data-baseweb="menu"] *,
    ul[role="listbox"] *,
    div[role="listbox"] * {
        color: #111827 !important;
    }
    
    /* Override any dark backgrounds in dropdowns */
    [data-baseweb="popup"][style*="background-color"],
    [data-baseweb="menu"][style*="background-color"],
    [role="listbox"][style*="background-color"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix BaseWeb select component specifically */
    [data-baseweb="select"] {
        background: #FFFFFF !important;
    }
    
    [data-baseweb="select"] [data-baseweb="select"] {
        background: #FFFFFF !important;
        color: #111827 !important;
    }
    
    /* Fix any nested dropdown elements */
    [data-baseweb="select"] [data-baseweb="popup"],
    [data-baseweb="select"] [data-baseweb="menu"] {
        background: #FFFFFF !important;
    }
    
    [data-baseweb="select"] [data-baseweb="popup"] *,
    [data-baseweb="select"] [data-baseweb="menu"] * {
        color: #111827 !important;
        background: transparent !important;
    }
    
    /* Ensure dropdowns in Streamlit containers are visible */
    .stSelectbox [data-baseweb="popup"],
    .stSelectbox [data-baseweb="menu"],
    .stMultiSelect [data-baseweb="popup"],
    .stMultiSelect [data-baseweb="menu"] {
        background: #FFFFFF !important;
        z-index: 9999 !important;
    }
    
    .stSelectbox [data-baseweb="popup"] *,
    .stSelectbox [data-baseweb="menu"] *,
    .stMultiSelect [data-baseweb="popup"] *,
    .stMultiSelect [data-baseweb="menu"] * {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8010"

# Session State Initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'eda_generated' not in st.session_state:
    st.session_state.eda_generated = False


def display_header():
    """Display the professional header"""
    st.markdown("""
    <div class="euron-header">
        <div class="euron-logo">
            <div class="euron-logo-icon">E</div>
            <div>
                <h1 class="euron-title"> ML Platform</h1>
                <p class="euron-subtitle">Enterprise Machine Learning Automation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_metric_card(label, value, is_highlighted=False):
    """Display a professional metric card"""
    value_class = "metric-value-blue" if is_highlighted else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {value_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def upload_data():
    """Handle data upload"""
    st.markdown('<div class="section-header">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6B7280; font-size: 14px; margin-bottom: 1rem;">Supported formats: CSV, Excel (XLS/XLSX), JSON</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload your dataset for automated ML analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your data..."):
            try:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data_info = response.json()
                    st.session_state.session_id = data_info['session_id']
                    st.session_state.data_info = data_info
                    
                    uploaded_file.seek(0)
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                        st.session_state.df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        st.session_state.df = pd.read_json(uploaded_file)
                    
                    st.success("Dataset uploaded successfully")
                    st.rerun()
                else:
                    st.error(f"Error: {response.json().get('detail', 'Upload failed')}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")


def display_data_overview():
    """Display data overview and statistics"""
    if st.session_state.data_info is None:
        return
    
    info = st.session_state.data_info
    df = st.session_state.df
    
    st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card("Total Rows", f"{info['rows']:,}", True)
    with col2:
        display_metric_card("Total Columns", info['columns'])
    with col3:
        display_metric_card("Numeric Features", len(info['numeric_columns']))
    with col4:
        display_metric_card("Categorical Features", len(info['categorical_columns']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Preview Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Missing Values", "Correlations"])
    
    with tab1:
        st.markdown("**First 10 Rows**")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        
        with st.expander("View Last 10 Rows"):
            st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(3), use_container_width=True)
        
        st.markdown("**Column Information**")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    
    with tab3:
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing Count', ascending=False)
        
        if missing_data.sum() > 0:
            fig = px.bar(
                missing_df[missing_df['Missing Count'] > 0],
                x='Column',
                y='Missing %',
                color='Missing %',
                color_continuous_scale=[[0, '#E5E7EB'], [1, '#B91C1C']],
                title='Missing Values by Column'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter", color="#111827"),
                title_font=dict(size=16, color="#111827"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the dataset")
        
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    with tab4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale=[[0, '#B91C1C'], [0.5, '#FFFFFF'], [1, '#0A66C2']],
                aspect='auto',
                title='Correlation Matrix'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter", color="#111827"),
                title_font=dict(size=16, color="#111827")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")


def generate_eda_report():
    """Generate and display EDA report"""
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="euron-card">
            <p style="color: #111827; font-size: 14px; margin-bottom: 0.5rem;">Generate a comprehensive EDA report including:</p>
            <ul style="color: #6B7280; font-size: 14px; margin: 0; padding-left: 1.5rem;">
                <li>Dataset overview and alerts</li>
                <li>Variable distributions and statistics</li>
                <li>Correlation analysis</li>
                <li>Missing value analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Generate Report", use_container_width=True):
            with st.spinner("Generating EDA report..."):
                try:
                    response = requests.post(
                        f"{API_URL}/data/{st.session_state.session_id}/eda-report"
                    )
                    if response.status_code == 200:
                        st.session_state.eda_generated = True
                        st.success("EDA Report generated successfully")
                        st.rerun()
                    else:
                        st.error(f"Error generating report: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.eda_generated:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**EDA Report**")
        report_url = f"{API_URL}/data/{st.session_state.session_id}/eda-report"
        st.markdown(f"""
        <iframe src="{report_url}" width="100%" height="700px" style="border: 1px solid #E5E7EB; border-radius: 8px;"></iframe>
        """, unsafe_allow_html=True)


def feature_selection():
    """Handle feature and target selection"""
    st.markdown('<div class="section-header">Feature Selection</div>', unsafe_allow_html=True)
    
    if st.session_state.data_info is None:
        st.warning("Please upload data first")
        return
    
    info = st.session_state.data_info
    df = st.session_state.df
    
    # Problem type selection
    st.markdown("**Problem Type**")
    problem_type = st.selectbox(
        "Select the type of ML problem",
        options=['classification', 'regression', 'clustering'],
        help="Choose the type of ML problem you want to solve",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if problem_type in ['classification', 'regression']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Features (X)**")
            all_columns = info['column_names']
            features = st.multiselect(
                "Select feature columns",
                options=all_columns,
                default=[col for col in all_columns if col != all_columns[-1]],
                help="Select the columns to use as input features",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Target Variable (y)**")
            remaining_cols = [col for col in all_columns if col not in features]
            target = st.selectbox(
                "Select target column",
                options=remaining_cols if remaining_cols else all_columns,
                help="Select the column to predict",
                label_visibility="collapsed"
            )
            
            if target:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Target Distribution**")
                if df[target].dtype == 'object' or df[target].nunique() <= 20:
                    fig = px.pie(
                        df, 
                        names=target, 
                        hole=0.4,
                        color_discrete_sequence=['#0A66C2', '#004182', '#3B82F6', '#60A5FA', '#93C5FD']
                    )
                else:
                    fig = px.histogram(
                        df, 
                        x=target,
                        nbins=30,
                        color_discrete_sequence=['#0A66C2']
                    )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter", color="#111827"),
                    height=250,
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Clustering
        st.markdown("**Features for Clustering**")
        all_columns = info['column_names']
        features = st.multiselect(
            "Select feature columns",
            options=all_columns,
            default=info['numeric_columns'],
            help="Select the columns to use for clustering",
            label_visibility="collapsed"
        )
        target = None
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Number of Clusters**")
        n_clusters = st.slider("For KMeans & Agglomerative", 2, 10, 3, label_visibility="collapsed")
        st.session_state.n_clusters = n_clusters
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Confirm Selection", use_container_width=True):
        if not features:
            st.error("Please select at least one feature")
            return
        
        if problem_type != 'clustering' and not target:
            st.error("Please select a target column")
            return
        
        try:
            response = requests.post(
                f"{API_URL}/select-features",
                json={
                    'session_id': st.session_state.session_id,
                    'features': features,
                    'target': target,
                    'problem_type': problem_type
                }
            )
            
            if response.status_code == 200:
                st.session_state.features = features
                st.session_state.target = target
                st.session_state.problem_type = problem_type
                st.success("Features selected successfully")
                
                result = response.json()
                if 'detected_problem_type' in result:
                    if result['detected_problem_type'] != problem_type:
                        st.info(f"Auto-detected problem type: {result['detected_problem_type']} (You selected: {problem_type})")
            else:
                st.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def model_training():
    """Handle model training and comparison"""
    st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'features') or st.session_state.features is None:
        st.warning("Please select features first")
        return
    
    # Get available models
    try:
        response = requests.get(f"{API_URL}/models/available")
        available_models = response.json()
    except:
        available_models = {
            'classification': ['Logistic Regression', 'SVM Classifier', 'Decision Tree Classifier', 
                             'Random Forest Classifier', 'KNN Classifier', 'Gradient Boosting Classifier', 'XGBoost Classifier'],
            'regression': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR',
                          'Decision Tree Regressor', 'Random Forest Regressor', 'KNN Regressor', 
                          'Gradient Boosting Regressor', 'XGBoost Regressor'],
            'clustering': ['KMeans', 'DBSCAN', 'Agglomerative Clustering']
        }
    
    problem_type = st.session_state.problem_type
    
    # Training Configuration
    st.markdown("**Training Configuration**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data for testing"
        )
    
    with col2:
        handle_missing = st.selectbox(
            "Handle Missing Values",
            options=['mean', 'median', 'mode', 'drop'],
            help="Strategy for missing values"
        )
    
    with col3:
        random_state = st.number_input("Random State", value=42, help="Seed for reproducibility")
    
    scale_features = st.checkbox("Scale Features", value=True, help="Standardize features before training")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Selection
    st.markdown("**Select Models**")
    model_options = available_models.get(problem_type, [])
    
    col1, col2 = st.columns([4, 1])
    with col1:
        selected_models = st.multiselect(
            "Choose models to train",
            options=model_options,
            default=model_options,
            help="Select which models to train and compare",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Select All"):
            selected_models = model_options
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Train button
    if st.button("Train Models", use_container_width=True, type="primary"):
        if not selected_models:
            st.error("Please select at least one model")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Training models..."):
            try:
                status_text.text("Preprocessing data...")
                progress_bar.progress(20)
                
                response = requests.post(
                    f"{API_URL}/train",
                    json={
                        'session_id': st.session_state.session_id,
                        'test_size': test_size,
                        'random_state': int(random_state),
                        'scale_features': scale_features,
                        'handle_missing': handle_missing,
                        'selected_models': selected_models
                    },
                    timeout=300
                )
                
                progress_bar.progress(80)
                status_text.text("Processing results...")
                
                if response.status_code == 200:
                    results = response.json()
                    st.session_state.training_results = results['results']
                    st.session_state.best_model = results['best_model']
                    progress_bar.progress(100)
                    status_text.text("Training complete")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except requests.exceptions.JSONDecodeError as e:
                st.error(f"Error parsing response: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def display_results():
    """Display training results and comparisons"""
    if st.session_state.training_results is None:
        return
    
    results = st.session_state.training_results
    best_model = st.session_state.best_model
    problem_type = st.session_state.problem_type
    
    st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)
    
    # Best Model Card
    if best_model and best_model.get('name'):
        metric_name = 'accuracy' if problem_type == 'classification' else 'r2_score' if problem_type == 'regression' else 'silhouette_score'
        metric_value = best_model['metrics'].get(metric_name, 0)
        
        st.markdown(f"""
        <div class="best-model-card">
            <span class="best-model-badge">Best Model</span>
            <div class="best-model-name">{best_model['name']}</div>
            <p style="color: #6B7280; font-size: 14px; margin: 0;">
                {metric_name.replace('_', ' ').title()}: <strong style="color: #0A66C2;">{metric_value:.4f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Best model metrics
        if best_model.get('metrics'):
            cols = st.columns(len(best_model['metrics']))
            for i, (metric, value) in enumerate(best_model['metrics'].items()):
                with cols[i]:
                    display_metric_card(metric.replace('_', ' ').title(), f"{value:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Comparison Charts
    st.markdown("**Model Comparison**")
    
    # Prepare comparison data
    comparison_data = []
    for result in results:
        if 'error' not in result and result.get('metrics'):
            row = {'Model': result['model_name'], 'Training Time (s)': result['training_time']}
            row.update(result['metrics'])
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Determine main metric based on problem type
        if problem_type == 'classification':
            main_metric = 'accuracy'
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        elif problem_type == 'regression':
            main_metric = 'r2_score'
            metrics_to_plot = ['r2_score', 'rmse', 'mae']
        else:
            main_metric = 'silhouette_score'
            metrics_to_plot = ['silhouette_score', 'calinski_harabasz_score']
        
        # Bar chart comparison
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Performance Metrics', 'Training Time (seconds)'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#0A66C2', '#004182', '#3B82F6', '#60A5FA']
        
        for i, metric in enumerate(available_metrics):
            fig.add_trace(
                go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Bar(
                name='Training Time',
                x=comparison_df['Model'],
                y=comparison_df['Training Time (s)'],
                marker_color='#6B7280',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", color="#111827"),
            height=400,
            barmode='group',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown('<h3 style="color: #111827 !important; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem;">Detailed Metrics</h3>', unsafe_allow_html=True)
        display_df = comparison_df.copy()
        for col in display_df.columns:
            if col not in ['Model']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Individual model details
        st.markdown('<h3 style="color: #111827 !important; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem;">Individual Model Details</h3>', unsafe_allow_html=True)
        
        for result in results:
            if 'error' in result:
                with st.expander(f"{result['model_name']} - Error"):
                    st.error(result['error'])
            else:
                with st.expander(f"{result['model_name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<p style="color: #111827 !important; font-weight: 600; margin-bottom: 0.5rem;"><strong>Metrics</strong></p>', unsafe_allow_html=True)
                        for metric, value in result['metrics'].items():
                            st.markdown(f'<p style="color: #111827 !important; margin: 0.25rem 0;">{metric.replace("_", " ").title()}: <code style="background: #F3F6F8; color: #0A66C2; padding: 2px 6px; border-radius: 4px;">{value:.4f}</code></p>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<p style="color: #111827 !important; font-weight: 600; margin-bottom: 0.5rem;"><strong>Training Info</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: #111827 !important; margin: 0.25rem 0;">Training Time: <code style="background: #F3F6F8; color: #0A66C2; padding: 2px 6px; border-radius: 4px;">{result["training_time"]:.4f}s</code></p>', unsafe_allow_html=True)
                        if 'cv_mean' in result['metrics']:
                            st.markdown(f'<p style="color: #111827 !important; margin: 0.25rem 0;">CV Mean: <code style="background: #F3F6F8; color: #0A66C2; padding: 2px 6px; border-radius: 4px;">{result["metrics"]["cv_mean"]:.4f}</code></p>', unsafe_allow_html=True)
                            st.markdown(f'<p style="color: #111827 !important; margin: 0.25rem 0;">CV Std: <code style="background: #F3F6F8; color: #0A66C2; padding: 2px 6px; border-radius: 4px;">{result["metrics"]["cv_std"]:.4f}</code></p>', unsafe_allow_html=True)
                    
                    if 'confusion_matrix' in result:
                        st.markdown('<p style="color: #111827 !important; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;"><strong>Confusion Matrix</strong></p>', unsafe_allow_html=True)
                        cm = np.array(result['confusion_matrix'])
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale=[[0, '#F3F6F8'], [1, '#0A66C2']],
                            labels=dict(x="Predicted", y="Actual", color="Count")
                        )
                        fig_cm.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Inter", color="#111827", size=12),
                            title_font=dict(family="Inter", color="#111827", size=14),
                            height=300
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)


def download_model():
    """Handle model download"""
    st.markdown('<div class="section-header">Download Model</div>', unsafe_allow_html=True)
    
    if st.session_state.best_model is None or not st.session_state.best_model.get('name'):
        st.warning("No trained model available. Please train models first.")
        return
    
    best_model = st.session_state.best_model
    
    st.markdown(f"""
    <div class="euron-card">
        <div class="euron-card-header">Best Model: {best_model['name']}</div>
        <p style="color: #6B7280; font-size: 14px;">Download the trained model as a pickle file for deployment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Model (.pkl)", use_container_width=True):
            try:
                response = requests.get(
                    f"{API_URL}/models/{st.session_state.session_id}/download"
                )
                if response.status_code == 200:
                    st.download_button(
                        label="Save Model File",
                        data=response.content,
                        file_name=f"euron_model_{st.session_state.session_id[:8]}.pkl",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("Error downloading model")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("""
        **Model includes:**
        - Trained model object
        - Feature scaler
        - Label encoders
        - Feature names
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Usage Example**")
    st.markdown("""
    <div style="background: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <pre style="color: #F3F6F8; font-family: 'JetBrains Mono', 'Courier New', monospace; margin: 0; white-space: pre-wrap;"><code style="color: #F3F6F8;">import pickle

# Load the model
with open('euron_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# Make predictions
# X_new = your_new_data[features]
# if scaler:
#     X_new = scaler.transform(X_new)
# predictions = model.predict(X_new)</code></pre>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0 0.5rem;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 1.5rem;">
                <div style="width: 32px; height: 32px; background: #0A66C2; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 16px;">E</div>
                <div>
                    <div style="font-weight: 600; color: #111827; font-size: 14px;"> ML</div>
                    <div style="font-size: 11px; color: #6B7280;">v1.0.0</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown('<p style="font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">Navigation</p>', unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            options=[
                "Upload Data",
                "Data Overview",
                "EDA Report",
                "Feature Selection",
                "Train Models",
                "Results",
                "Download Model",
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Session info
        if st.session_state.session_id:
            st.markdown('<p style="font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">Session Info</p>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="font-size: 13px; color: #111827;">
                <p style="margin: 0.25rem 0;"><strong>Session:</strong> {st.session_state.session_id[:8]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.data_info:
                st.markdown(f"""
                <div style="font-size: 13px; color: #111827;">
                    <p style="margin: 0.25rem 0;"><strong>Dataset:</strong> {st.session_state.data_info['filename']}</p>
                    <p style="margin: 0.25rem 0;"><strong>Shape:</strong> {st.session_state.data_info['rows']} x {st.session_state.data_info['columns']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
                st.markdown(f"""
                <div style="font-size: 13px; color: #111827;">
                    <p style="margin: 0.25rem 0;"><strong>Type:</strong> {st.session_state.problem_type.title()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Reset Session", use_container_width=True):
                try:
                    requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
                except:
                    pass
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content based on navigation
    if page == "Upload Data":
        upload_data()
        if st.session_state.data_info:
            display_data_overview()
    
    elif page == "Data Overview":
        if st.session_state.data_info:
            display_data_overview()
        else:
            st.warning("Please upload data first")
    
    elif page == "EDA Report":
        if st.session_state.data_info:
            generate_eda_report()
        else:
            st.warning("Please upload data first")
    
    elif page == "Feature Selection":
        feature_selection()
    
    elif page == "Train Models":
        model_training()
        if st.session_state.training_results:
            display_results()
    
    elif page == "Results":
        if st.session_state.training_results:
            display_results()
        else:
            st.warning("Please train models first")
    
    elif page == "Download Model":
        download_model()
    


if __name__ == "__main__":
    main()
