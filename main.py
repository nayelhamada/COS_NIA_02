# Modules standard
from datetime import datetime
import streamlit as st
import os
import json
import streamlit_mermaid as st_mermaid
from streamlit_calendar import calendar
import signal
import subprocess
import time
import sys

# Nos modules
import utils
from modules import auth, files, rag, generators, schedule

# 1. Configuration Initiale
utils.init_folders()
utils.setup_page(st)

# Apparence des tableaux et boutons
def inject_custom_css():
    st.markdown("""
    <style>
        table { width: 100%; border-collapse: separate; border-radius: 10px; overflow: hidden; border: 1px solid #4A4A4A; background-color: #1E1E1E; color: #E0E0E0; font-family: 'Courier New', Courier, monospace; }
        thead tr th { background: linear-gradient(90deg, #2b5876 0%, #4e4376 100%); color: #ffffff; font-weight: bold; text-transform: uppercase; padding: 15px; border-bottom: 2px solid #FFD700; }
        tbody tr td { padding: 12px; border-bottom: 1px solid #333333; }
        tbody tr:hover { background-color: #2C2C2C; cursor: default; }
        td:first-child { font-weight: bold; color: #4db8ff; }
        .stButton button { width: 100%; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# 2. Gestion Session
if "user_session" not in st.session_state: st.session_state["user_session"] = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR (LOGIN & PROFIL) ---
with st.sidebar:
    st.header("🔒 Login")
    
    # cas 1 : utilisateur connecté
    if st.session_state["user_session"]:
        user_info = st.session_state["user_session"]
        username = user_info['auth']['username_display']
        display_role = user_info["profil"].get("role", "Astronaut")
        st.success(f"{display_role} : **{username}**")
        
        # --- BLOC MODIFIER MON PROFIL ---
        with st.expander("👤 Your information"):
            st.caption("See and update your profile details and AI preferences.")
            
            # Récupération des données
            raw_level = user_info["profil"].get("niveau", "Average")
            raw_tone = user_info["profil"].get("preferences_apprentissage", {}).get("ton", "Neutral")
            raw_role = user_info["profil"].get("role", "Cadet")
            raw_goal = user_info["profil"].get("objectif", "")

            # définir listes d'options
            list_niveaux = ["Beginner", "Average", "Advanced", "Expert"]
            list_styles = ["Cool", "Strict", "Neutral", "Scientific", "Military"]
            list_roles = ["Cadet", "Pilot", "Engineer", "Scientist", "Capitain", "Tourist"]

            # Gestions des valeurs par défaut (en cas de données corrompues ou manquantes)
            # Niveau         
            default_level = raw_level if raw_level in list_niveaux else "Average"

            # Style
            default_tone = raw_tone if raw_tone in list_styles else "Neutral"
            
            # Rôle
            default_role = raw_role if raw_role in list_roles else "Cadet"

            # Formulaire de mise à jour
            with st.form("update_profile_full"):
                st.markdown("**Identity**")
                new_role = st.selectbox("Speciality", list_roles, index=list_roles.index(default_role))
                new_goal = st.text_input("Mission Objective", value=raw_goal, placeholder="Ex: Assistant")
                
                st.markdown("**AI Preferences**")
                new_level = st.select_slider("Expertise Level", options=list_niveaux, value=default_level, key="level_slider_main")
                new_tone = st.selectbox("AI Tone", list_styles, index=list_styles.index(default_tone))
                
                if st.form_submit_button("💾 Update Profile"):
                    updated_data = auth.update_user_profile(username, new_level, new_tone, new_role, new_goal)
                    if updated_data:
                        st.session_state["user_session"] = updated_data
                        st.toast("Profile updated!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Error updating profile.")

        st.divider()
        
        # --- BLOC SÉLECTION DU MODE ---
        st.header("⚙️ Output Format")
        selected_mode = st.radio(
            "Output Format :",
            list(rag.PROMPT_MODES.keys()),
            key="mode_radio_main" 
        )
        
        st.divider()
        
        if st.button("Logout", key="logout_btn_main"):
            st.session_state["user_session"] = None
            st.rerun()
            
        st.divider()
        
        uploaded = st.file_uploader("Add PDF", type="pdf", key="pdf_uploader_main")
        if uploaded:
            save_path = os.path.join(utils.COURS_FOLDER, uploaded.name)
            with open(save_path, "wb") as f: 
                f.write(uploaded.getbuffer())
            st.success("Document archivé !")

    # cas 2 : pas de session, afficher formulaire de connexion / inscription
    else:
        t1, t2 = st.tabs(["Log in", "Sign Up"])
        
        with t1:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Log in"):
                    d, m = auth.verify_credentials(u, p)
                    if d: 
                        st.session_state["user_session"] = d
                        st.rerun()
                    else: 
                        st.error(m)
        
        with t2:
            st.markdown("Join **Spaceflight Institute**")
            with st.form("signup_form"):
                u = st.text_input("New Username")
                p = st.text_input("Create Password", type="password")
                
                c1, c2 = st.columns(2)
                with c1:
                    role_signup = st.selectbox("Specialty", ["Cadet", "Pilot", "Engineer", "Scientist", "Commander", "Tourist"])
                with c2:
                    level_signup = st.selectbox("Expertise Level", ["Beginner", "Intermediate", "Advanced", "Expert"])
                
                goal_signup = st.text_input("Mission Objective", placeholder="Ex: Apprendre la physique orbitale")
                tone_signup = st.selectbox("AI Tone", ["Cool", "Strict", "Neutral", "Scientific", "Military"])
                
                if st.form_submit_button("Initialize Profile"):
                    created, msg = auth.create_user(u, p, level_signup, tone_signup, role_signup, goal_signup)
                    if created:
                        st.success("Profile created! Please log in.")
                    else:
                        st.error(msg)
    
    st.divider()
    if st.button("🛑 Stop system"):
        st.warning("Forcing shutdown of Streamlit and Ollama...")
        
        # Script AppleScript avec mise au premier plan (activate)
        applescript = """
        ignoring application responses
            
            -- 1. FERMER CHROME
            tell application "Google Chrome"
                try
                    close (every tab of every window whose URL contains ":8501")
                end try
            end tell
            
            -- 2. GESTION OLLAMA
            try
                do shell script "pkill -9 -f ollama"
            end try
            
            delay 0.2
            
            -- On met le Terminal au premier plan pour voir la fenêtre se fermer
            tell application "Terminal"
                activate  -- <--- C'EST CETTE COMMANDE QUI FAIT LA MAGIE
                try
                    close (every window whose name contains "ollama")
                end try
            end tell

            -- 3. GESTION STREAMLIT
            try
                do shell script "pkill -9 -f streamlit"
            end try
            
            delay 0.2
            
            -- Idem ici, on s'assure que le Terminal est devant
            tell application "Terminal"
                activate -- <--- ON LE RÉACTIVE ICI
                try
                    close (every window whose name contains "main.py" or name contains "streamlit")
                end try
            end tell
            
        end ignoring
        """

        # Exécution
        subprocess.Popen(["osascript", "-e", applescript])
        
        time.sleep(0.5)
        st.stop()

# --- MAIN CHAT ---
if not st.session_state["user_session"]: st.stop()

for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # affichage du calendrier si la clé existe dans le message (indiquant que c'est une réponse de planning)
        if "calendar_data" in m:
            cal_opts = {
                "editable": False, "headerToolbar": {"left": "", "center": "title", "right": ""},
                "firstDay": 1, "initialView": "timeGridWeek", "initialDate": "2024-01-01",
                "slotMinTime": "08:00:00", "slotMaxTime": "18:00:00", "allDaySlot": False,
                "locale": "fr", "height": "auto", "weekends": False
            }
            # Widget Calendrier
            calendar(events=m["calendar_data"], options=cal_opts, key=f"cal_{idx}")
            
            # Bouton PDF
            try:
                username = user_info['auth']['username_display']
                pdf_bytes = schedule.create_planning_pdf(m["calendar_data"], username)
                st.download_button(
                    label="📄 Télécharger le Planning (PDF)",
                    data=pdf_bytes, file_name=f"Planning_{username}.pdf", mime="application/pdf", key=f"pdf_btn_{idx}"
                )
            except Exception as e:
                st.error(f"Erreur PDF: {e}")

            # Détails en liste (est ce que c'est vraiment utile ?)
            with st.expander("📋 Liste détaillée"):
                revs = [e for e in m["calendar_data"] if e.get("backgroundColor") in ["#0984e3", "#6c5ce7"]]
                revs.sort(key=lambda x: x['start'])
                days = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi", "Thursday": "Jeudi", "Friday": "Vendredi"}
                for r in revs:
                    s = datetime.strptime(r['start'], "%Y-%m-%dT%H:%M:%S")
                    e_dt = datetime.strptime(r['end'], "%Y-%m-%dT%H:%M:%S")
                    d_fr = days.get(s.strftime("%A"), s.strftime("%A"))
                    st.markdown(f"**{d_fr} {s.strftime('%H:%M')} - {e_dt.strftime('%H:%M')}** : {r['extendedProps']['fullTitle']}")

if prompt := st.chat_input("Your question or command..."):

    # Affichage User
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        
        # Interception des mots-clés du planning 
        mots_cles_planning = ["plan", "agenda", "emploi du temps"]
        if any(kw in prompt.lower() for kw in mots_cles_planning):
            st.caption("🗓️ Generating schedule...")
            
            # générqtion du calendrier
            sch = user_info.get("utilisateur", {}).get("disponibilites", {})
            raw_csv = schedule.generate_revision_plan(user_info, utils.COURS_FOLDER)
            events = schedule.parse_schedule_to_events(raw_csv, sch)
            if isinstance(events, set): events = list(events)
            
            # Sauvegarde avec les données du calendrier
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "✅ Voici votre planning optimisé (Lundi-Vendredi, 08h-18h) :",
                "calendar_data": events
            })
            
            # On force le rechargement de la page pour afficher le beau calendrier (qui est en fait un truc random)
            st.rerun()


        else:
            # Contextualisation 
            real_prompt = prompt
            if len(st.session_state.messages) > 1:
                with st.status("🧠 Analyse du contexte...", expanded=False) as status:
                    history = st.session_state.messages[:-1]
                    new_prompt = rag.contextualize_question(prompt, history) #On utilise l'IA pour générer une question reformulée qui intègre le contexte de la conversation
                    if new_prompt != prompt:
                        real_prompt = new_prompt
                        status.write(f"Nex prompt : {real_prompt}")
                        status.update(label="Contexte loaded", state="complete")
                    else:
                        status.update(label="No necessary context.", state="complete")

            # Smart Router, si l'utilisateur mentionne un fichier ou un cours spécifique, on lui propose directement le téléchargement avant même de lancer le RAG 
            file_path = files.smart_file_router(real_prompt, utils.COURS_FOLDER)
            if file_path:
                fname = os.path.basename(file_path)
                st.success(f"Document found : {fname}")
                with open(file_path, "rb") as f:
                    st.download_button("download", f, file_name=fname)
                real_prompt += f" (Note: Fichier {fname} proposé.)"

            # RAG 
            rel_files, _ = files.get_relevant_files(real_prompt, utils.COURS_FOLDER) # On récupère les fichiers pertinents à la question de l'utilisateur, pour les injecter dans le RAG et améliorer la réponse. Si aucun fichier pertinent, on injecte tout le dossier
            if rel_files:
                with st.spinner("Analysis..."):
                    custom_instr = rag.PROMPT_MODES[selected_mode] if selected_mode != "Chat Standard" else None
                    chain = rag.initialize_rag_chain_dynamic(rel_files, st.session_state["user_session"], custom_instr)
                    
                    if chain:
                        res = chain.invoke({"input": real_prompt})
                        raw = res["answer"]
                        
                        # Nettoyage des éléments générés inutiles
                        clean_txt = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
                        final_content = raw
                        if "<tool_call>" in raw:
                            final_content = raw.split("<tool_call>")[-1].strip() 
                        final_content_clean = final_content.replace("```json", "").replace("```mermaid", "").replace("```", "").strip()


                        # Affichage et génération de contenu selon le mode sélectionné
                        if selected_mode == "Chat Standard":
                            st.markdown(final_content)
                        
                        elif selected_mode == "🎙️ Audio (Podcast)":
                            st.markdown("### 🎙️ Podcast")
                            with st.expander("Script"): st.write(final_content)
                            audio_file = generators.generate_audio(final_content)
                            if audio_file: st.audio(audio_file)
                        
                        elif selected_mode == "🧠 visual card":
                            st.markdown("### 🧠 Carte Mentale")
                            mermaid_code = generators.clean_mermaid_code(final_content)
                            try:
                                st_mermaid.st_mermaid(mermaid_code, height="500px")
                            except:
                                st.code(mermaid_code)
                        
                        elif selected_mode == "📝 Flash card":
                            st.markdown("### 📝 Flashcards")
                            try:
                                flashcards = json.loads(final_content_clean)
                                cols = st.columns(2)
                                for i, card in enumerate(flashcards):
                                    with cols[i % 2]:
                                        with st.expander(f"❓ {card.get('question', 'Q')}", expanded=False):
                                            st.info(f"💡 {card.get('reponse', 'R')}")
                            except:
                                st.warning("Erreur JSON.")
                                st.write(final_content)

                        elif selected_mode == "📊 Slides (PPTX)":
                            st.markdown("### 📊 PowerPoint")
                            try:
                                slides_data = json.loads(final_content_clean)
                                pptx_file = generators.generate_pptx_from_json(slides_data)
                                with open(pptx_file, "rb") as f:
                                    st.download_button("⬇️ Télécharger .pptx", f, "cours.pptx")
                                for slide in slides_data:
                                    st.markdown(f"**📺 {slide.get('titre', 'Slide')}**")
                                    for p in slide.get('points', []):
                                        st.markdown(f"- {p}")
                            except:
                                st.warning("Erreur PPTX.")
                                st.write(final_content)

                        # Sauvegarde historique (Réponse nette)
                        st.session_state.messages.append({"role": "assistant", "content": final_content})