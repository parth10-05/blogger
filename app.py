import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import time
import random

class BlogGenerator:
    def __init__(self, groq_api_key, model_name="llama3-70b-8192"):
        # Initialize Groq LLM with valid model names
        self.llm = ChatGroq(
            temperature=0.7,
            model_name=model_name,
            groq_api_key=groq_api_key
        )
        
    def research_content(self, topic):
        """Research content and gather relevant information about the topic"""
        current_date = datetime.now().strftime("%Y-%m-%d")

        prompt = PromptTemplate(
            input_variables=["topic", "current_date"],
            template="""
            As a professional research assistant, gather comprehensive and CURRENT information about: {topic}

            Today's date is {current_date}. Prioritize information from the last 6 months.

            Provide:
            1. Key facts and statistics (with sources cited as clickable links)
            2. Current trends and developments (with clickable sources where applicable)
            3. Common misconceptions
            4. Expert opinions or quotes (with attribution)
            5. Related subtopics worth exploring

            FOR CURRENT INFORMATION:
            - Clearly indicate when information was published (month and year)
            - For each fact/statistic, include the publication date
            - If data is older than 6 months, flag it as potentially outdated
            - Format all sources as clickable markdown links: [Source Name](URL)

            Example format:
            - "According to recent data (March 2024), 72% of users prefer mobile apps [Source: Pew Research](https://www.pewresearch.org)"

            Format your response with clear headings for each section.
            Include markdown formatting for better readability.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(topic=topic, current_date=current_date)
        return result
    
    def generate_titles(self, topic):
        """Generate blog title suggestions based on topic"""
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            You're an expert content strategist. Suggest 5 engaging blog title options about {topic}.
            The titles should be:
            - SEO-friendly
            - Appealing to readers
            - Reflective of current trends (mention if relevant)
            
            Format your response as a numbered list with no additional commentary.
            Do not add the response line of here are five titles.

            Example format:
            1. Title One [Trending in 2025]
            2. Title Two
            3. Title Three
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(topic=topic)
        return result
    
    def generate_keywords(self, title):
        """Suggest relevant keywords for the blog"""
        current_year = datetime.now().year
        
        prompt = PromptTemplate(
            input_variables=["title", "current_year"],
            template="""
            Suggest 10-15 relevant keywords and important concepts that should be included 
            in a blog post titled: {title}

            Do not add any additional commentary like Here is the list of keywords and concepts:

            Include:
            - Current year ({current_year}) where relevant
            - Trending terms related to the topic
            - Long-tail keywords
            
            Format as a comma-separated list with no additional commentary.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(title=title, current_year=current_year)
        return result
    
    def generate_blog(self, title, keywords, word_limit):
        """Generate full blog content based on selected title and keywords"""
        current_date = datetime.now().strftime("%B %Y")

        prompt = PromptTemplate(
            input_variables=["title", "keywords", "word_limit", "current_date"],
            template="""
            Write a comprehensive, SEO-optimized blog post with the following details:

            Title: {title}
            Keywords to include: {keywords}
            Word limit: Approximately {word_limit} words
            Current date: {current_date}

            Requirements:
            - Target approximately {word_limit} words
            - Use markdown formatting
            - Include headings (H2, H3) for proper structure
            - Write in a professional yet engaging tone
            - Include relevant examples where appropriate
            - End with a conclusion and call-to-action
            - Format all sources as clickable markdown links: [Source Name](URL)

            STRICT CURRENT INFORMATION REQUIREMENTS:
            - Clearly state the publication date for all facts/statistics
            - If using older data, explain why it's still relevant
            - Include at least 3 recent (last 6 months) references with clickable links
            - For time-sensitive topics, note when readers should check for updates

            Structure:
            # [Title]
            *Last updated: [Month Year]*

            ## Introduction
            [Engaging introduction paragraph mentioning current relevance]

            ## [Main Section 1]
            [Detailed content with clickable sources and dates where needed]

            ## [Main Section 2]
            [Detailed content with clickable sources and dates where needed]

            ## Conclusion
            [Summary and call-to-action with current relevance]

            ## References
            [List all clickable links used in the article]
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(title=title, keywords=keywords, word_limit=word_limit, current_date=current_date)
        return result
    
    def generate_qa(self, blog_content):
        """Generate Q&A section based on blog content"""
        current_date = datetime.now().strftime("%B %Y")
        
        prompt = PromptTemplate(
            input_variables=["blog_content", "current_date"],
            template="""
            Today's date is {current_date}. Based on the following blog content, generate a comprehensive Q&A section:
            
            {blog_content}
            
            Create 5-8 thoughtful questions a reader might have after reading this content,
            and provide detailed answers using information from the blog.
            
            For each answer:
            - Note how current the information is
            - If data is older than 6 months, suggest checking for updates
            - Include reference dates for all facts
            - Format all sources as clickable markdown links: [Source Name](URL)
            
            Format as:
            
            ## Frequently Asked Questions (Updated {current_date})
            
            ### [Question 1]
            [Answer 1 with date references and clickable links]
            
            ### [Question 2]
            [Answer 2 with date references and clickable links]
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(blog_content=blog_content, current_date=current_date)
        return result
    
    def initialize_chatbot(self):
        """Initialize the chatbot with memory"""
        memory = ConversationBufferMemory()
        return memory
    
    def chat_with_blog(self, memory, user_input, blog_content=None):
        """Chat with the AI about the blog content"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if blog_content:
            # Create a memory-aware chain specifically for blog content
            prompt = PromptTemplate(
                input_variables=["current_date", "blog_content", "history", "user_input"],
                template="""
                **Current Date**: {current_date}
                
                **Blog Content Reference**:
                {blog_content}
                
                **Conversation History**:
                {history}
                
                **User Question**: {user_input}
                
                **Instructions**:
                - Answer specifically about the referenced blog content
                - Include dates for any facts/statistics
                - Flag potentially outdated information
                - Format sources as [Source](URL)
                - Keep responses professional but conversational
                
                **Response**:
                """
            )

            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            
            # Prepare all inputs in a single dictionary
            inputs = {
                "current_date": current_date,
                "blog_content": blog_content[:8000],  # Truncate to prevent overflow
                "history": memory.buffer,
                "user_input": user_input
            }
            
            try:
                response = chain(inputs, return_only_outputs=True)
                return response['text']
            except Exception as e:
                return f"Error generating response: {str(e)}"
                
        else:
            # General conversation without blog content
            prompt = PromptTemplate(
                input_variables=["current_date", "history", "input"],
                template="""
                **Current Date**: {current_date}
                
                **Conversation History**:
                {history}
                
                **User Message**: {input}
                
                **Response**:
                """
            )

            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            
            inputs = {
                "current_date": current_date,
                "history": memory.buffer,
                "input": user_input
            }
            
            try:
                response = chain(inputs, return_only_outputs=True)
                return response['text']
            except Exception as e:
                return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Blog Generator Pro",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.title("Settings")
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
        st.markdown("[Get your Groq API key](https://console.groq.com/)")
        
        model_name = st.selectbox(
            "Select AI Model:",
            ["llama3-70b-8192", "llama-3.3-70b-versatile", "gemma-7b-it"],
            index=0
        )
        
        if groq_api_key:
            st.success("API key configured!")
        
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("- Up-to-date Research Assistant")
        st.markdown("- Current Title Generator")
        st.markdown("- Full Blog Generation")
        st.markdown("- Interactive Q&A Section")
        st.markdown("- Content Chatbot Assistant")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue")
        return
    
    # Initialize blog generator
    generator = BlogGenerator(groq_api_key, model_name)
    
    # Initialize session state variables
    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = ""
    if 'research_data' not in st.session_state:
        st.session_state.research_data = None
    if 'generated_blog' not in st.session_state:
        st.session_state.generated_blog = None
    if 'show_qa' not in st.session_state:
        st.session_state.show_qa = False
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = generator.initialize_chatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.title("AI Blog Generator Pro ✍️")
    st.write("Generate blog posts with current research and Q&A capabilities")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Research", "Content Creation", "Final Output", "Chat Assistant"])
    
    with tab1:
        st.header("Research Phase")
        topic = st.text_input("Enter your blog topic:", 
                            placeholder="e.g., EV industry")
        
        if topic:
            # Clear previous research if topic changes
            if 'previous_topic' in st.session_state and st.session_state.previous_topic != topic:
                st.session_state.research_data = None
                st.session_state.titles = None
                st.session_state.generated_blog = None
                st.session_state.selected_title = ""
            
            st.session_state.previous_topic = topic
            
            if st.button("Research Topic") or st.session_state.research_data:
                if not st.session_state.research_data:
                    with st.spinner("Gathering current research data..."):
                        try:
                            st.session_state.research_data = generator.research_content(topic)
                            # Add timestamp to research data
                            current_date = datetime.now().strftime("%Y-%m-%d")
                            st.session_state.research_data = f"# Research Report (Generated {current_date})\n\n" + st.session_state.research_data
                        except Exception as e:
                            st.error(f"An error occurred during research: {str(e)}")
                            st.session_state.research_data = "Research data unavailable due to technical issues. Please try again later."
                
                st.subheader("Research Findings")
                st.markdown(st.session_state.research_data)
                
                st.download_button(
                    label="Download Research Notes",
                    data=st.session_state.research_data,
                    file_name=f"{topic[:30]}_research_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

    with tab2:
        st.header("Content Creation")
        
        if not st.session_state.get('research_data'):
            st.info("Complete the Research phase first")
            st.stop()
            
        # Step 2: Generate title suggestions
        if st.button("Generate Title Suggestions") or 'titles' in st.session_state:
            with st.spinner("Generating current title suggestions..."):
                if 'titles' not in st.session_state:
                    st.session_state.titles = generator.generate_titles(st.session_state.previous_topic)
                
                st.subheader("Suggested Titles:")
                titles_list = [title.strip() for title in st.session_state.titles.split('\n') if title.strip()]
                for i, title in enumerate(titles_list, 1):
                    # Handle both "1. Title" and "Title" formats
                    title_text = title.split('. ', 1)[1] if '. ' in title else title
                    st.write(f"{i}. {title_text}")
        
        # Step 3: Let user select or enter a title
        title_options = st.radio(
            "Choose an option:",
            ("Select from suggestions", "Enter custom title"),
            horizontal=True,
            key="title_option"
        )
        
        if title_options == "Select from suggestions":
            if 'titles' in st.session_state:
                titles_list = [title.strip() for title in st.session_state.titles.split('\n') if title.strip()]
                title_options_list = [title.split('. ', 1)[1] if '. ' in title else title for title in titles_list]
                
                selected_title = st.selectbox(
                    "Select a title:",
                    options=title_options_list,
                    key="title_select"
                )
                st.session_state.selected_title = selected_title
            else:
                st.warning("Please generate title suggestions first")
        else:
            st.session_state.selected_title = st.text_input(
                "Enter your custom title:",
                key="custom_title",
                help="Include the current year if relevant (e.g., '2024')"
            )
        
        if st.session_state.selected_title:
            # Step 4: Keyword generation
            if st.button("Suggest Keywords") or 'suggested_keywords' in st.session_state:
                with st.spinner("Generating current keyword suggestions..."):
                    if 'suggested_keywords' not in st.session_state:
                        st.session_state.suggested_keywords = generator.generate_keywords(st.session_state.selected_title)
                    
                    st.subheader("Suggested Keywords:")
                    st.write(st.session_state.suggested_keywords)
                    st.info("You can copy these and edit as needed below")
            
            # Keyword input with suggested keywords as default
            default_keywords = st.session_state.get('suggested_keywords', '')
            keywords = st.text_area(
                "Enter keywords (comma-separated):",
                value=default_keywords,
                height=100,
                key="keywords_input",
                help="Include current year if relevant"
            )
            
            # Word limit input
            word_limit = st.slider(
                "Select word limit:",
                min_value=300,
                max_value=2000,
                value=800,
                step=100,
                key="word_limit"
            )
            
            # Step 5: Generate full blog
            if st.button("Generate Blog Post") and keywords:
                with st.spinner(f"Generating your {word_limit}-word blog post with current information..."):
                    try:
                        blog_content = generator.generate_blog(
                            st.session_state.selected_title, 
                            keywords,
                            word_limit
                        )
                        
                        # Add timestamp to generated blog
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        blog_content = f"<!-- Generated on {current_date} -->\n\n" + blog_content
                        
                        st.session_state.generated_blog = blog_content
                        st.session_state.show_qa = False  # Reset Q&A visibility
                        
                        st.subheader("Generated Blog Post")
                        st.markdown(blog_content)
                        
                        # Word count estimation
                        word_count = len(blog_content.split())
                        st.caption(f"Estimated word count: {word_count} words")
                    except Exception as e:
                        st.error(f"An error occurred while generating the blog: {str(e)}")

    with tab3:
        st.header("Final Output")
        
        if not st.session_state.get('generated_blog'):
            st.info("Generate a blog post in the Content Creation tab first")
            st.stop()
            
        st.markdown(st.session_state.generated_blog)
        
        # Q&A Generation Section
        if st.button("Generate Q&A Section"):
            with st.spinner("Creating comprehensive Q&A section with current information..."):
                try:
                    qa_content = generator.generate_qa(st.session_state.generated_blog)
                    st.session_state.generated_blog += "\n\n" + qa_content
                    st.session_state.show_qa = True
                except Exception as e:
                    st.error(f"An error occurred while generating Q&A: {str(e)}")
                
        if st.session_state.show_qa:
            st.markdown("---")
            st.subheader("Q&A Section")
            qa_content = generator.generate_qa(st.session_state.generated_blog)
            st.markdown(qa_content)
        
        # Final download button
        current_date = datetime.now().strftime("%Y%m%d")
        st.download_button(
            label="Download Full Content",
            data=st.session_state.generated_blog,
            file_name=f"{st.session_state.selected_title[:50].lower().replace(' ', '_')}_{current_date}.md",
            mime="text/markdown",
            key="final_download"
        )
    
    with tab4:
        st.header("Chat Assistant")
        st.write("Get current information and help with your blog content from our AI assistant")
        
        # Initialize chat if not already done
        if 'chat_memory' not in st.session_state:
            st.session_state.chat_memory = generator.initialize_chatbot()
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your blog..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get blog context if available
            blog_context = None
            if st.session_state.get('generated_blog'):
                blog_context = st.session_state.generated_blog
            
            # Generate AI response
            with st.spinner("Thinking..."):
                try:
                    response = generator.chat_with_blog(
                        st.session_state.chat_memory,
                        prompt,
                        blog_context
                    )
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Display AI response
                    with st.chat_message("assistant"):
                        st.markdown(response)
                except Exception as e:
                    st.error(f"An error occurred during chat: {str(e)}")
                            
        # Add some suggested prompts
        st.markdown("---")
        st.subheader("Try asking:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What's current in this field?"):
                if st.session_state.get('generated_blog'):
                    with st.spinner("Checking for current trends..."):
                        try:
                            response = generator.chat_with_blog(
                                st.session_state.chat_memory,
                                "What are the most current trends or developments related to this blog topic? Please include dates for any information.",
                                st.session_state.generated_blog
                            )
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please generate a blog post first")
            
            if st.button("Update statistics"):
                if st.session_state.get('generated_blog'):
                    with st.spinner("Finding current statistics..."):
                        try:
                            response = generator.chat_with_blog(
                                st.session_state.chat_memory,
                                "Can you provide the most current statistics relevant to this blog post? Include source dates.",
                                st.session_state.generated_blog
                            )
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please generate a blog post first")
        
        with col2:
            if st.button("Check for outdated info"):
                if st.session_state.get('generated_blog'):
                    with st.spinner("Analyzing content timeliness..."):
                        try:
                            response = generator.chat_with_blog(
                                st.session_state.chat_memory,
                                "Review this blog content and identify any information that might be outdated. For each item, suggest how to update it.",
                                st.session_state.generated_blog
                            )
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please generate a blog post first")
            
            if st.button("Recent case studies"):
                if st.session_state.get('generated_blog'):
                    with st.spinner("Finding recent examples..."):
                        try:
                            response = generator.chat_with_blog(
                                st.session_state.chat_memory,
                                "Can you suggest recent (last 6 months) case studies or examples relevant to this blog topic?",
                                st.session_state.generated_blog
                            )
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please generate a blog post first")

if __name__ == "__main__":
    main()
