// backend/server.js

require('dotenv').config(); // Load environment variables from .env file
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 3000; // Use port from environment variable or default to 3000

// Middleware
app.use(cors()); // Enable CORS for all origins (for development)
app.use(express.json({ limit: '10mb' })); // To parse JSON request bodies, increase limit for image data

// Initialize Gemini API
const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
    console.error("Error: GEMINI_API_KEY is not set in the .env file. Please check your .env file.");
    process.exit(1); // Exit if API key is missing
}
const genAI = new GoogleGenerativeAI(API_KEY);

// Choose a model for text-only operations (like translation, summarization)
const TEXT_MODEL_NAME = "gemini-1.5-flash"; // Or "gemini-pro" for older, "gemini-1.5-pro" for more capable
// Choose a model for multi-modal (vision) operations
const VISION_MODEL_NAME = "gemini-1.5-flash-001"; // Or "gemini-pro-vision"

// Safety settings for Gemini responses (optional but recommended)
const generationConfig = {
    temperature: 0.7, // Adjust creativity
    topP: 0.95,
    topK: 64,
    maxOutputTokens: 8192,
    responseMimeType: "text/plain", // Keep as text/plain for simplicity
};

const safetySettings = [
    {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
];

// --- API Endpoints ---

// 1. Language Translation
app.post('/api/translate', async (req, res) => {
    const { text, fromLang, toLang } = req.body;

    if (!text || !fromLang || !toLang) {
        return res.status(400).json({ error: 'Missing text, fromLang, or toLang in request.' });
    }

    try {
        const model = genAI.getGenerativeModel({ model: TEXT_MODEL_NAME });
        const prompt = `Translate the following text from ${fromLang} to ${toLang}. Only provide the translated text, nothing else.\nText: "${text}"`;

        const result = await model.generateContent(prompt, generationConfig, safetySettings);
        const response = await result.response;
        const translatedText = response.text().trim();

        res.json({ translatedText });

    } catch (error) {
        console.error('Error translating text:', error);
        // Log more details if available from Gemini API error
        if (error.response && error.response.data) {
            console.error('Gemini API Error details:', error.response.data);
        }
        res.status(500).json({ error: 'Failed to translate text. Please check your API key and input.', details: error.message });
    }
});

// 2. Text Summarization
app.post('/api/summarize', async (req, res) => {
    const { text, summaryType } = req.body;

    if (!text || !summaryType) {
        return res.status(400).json({ error: 'Missing text or summaryType in request.' });
    }

    let prompt = '';
    switch (summaryType) {
        case 'brief':
            prompt = `Provide a very brief, 1-2 sentence summary of the following text. Only the summary.\nText: "${text}"`;
            break;
        case 'detailed':
            prompt = `Provide a detailed, 3-5 sentence summary of the following text. Only the summary.\nText: "${text}"`;
            break;
        case 'bullets':
            prompt = `Summarize the following text into 3-5 key bullet points. Only the bullet points.\nText: "${text}"`;
            break;
        case 'key':
            prompt = `Extract the absolute 3-5 most important key points from the following text as a numbered list. Only the numbered list.\nText: "${text}"`;
            break;
        default:
            prompt = `Summarize the following text. Only the summary.\nText: "${text}"`;
            break;
    }

    try {
        const model = genAI.getGenerativeModel({ model: TEXT_MODEL_NAME });
        const result = await model.generateContent(prompt, generationConfig, safetySettings);
        const response = await result.response;
        const summary = response.text().trim();

        let keyPoints = '';
        // If the summary is already bullets/key points, use it for keyPoints
        if (summaryType === 'bullets' || summaryType === 'key') {
            keyPoints = summary;
        } else {
            // Otherwise, try to get separate key points
            const keyPointsPrompt = `Extract 3-5 key takeaways from the following text as bullet points:\n"${text}"`;
            const keyPointsResult = await model.generateContent(keyPointsPrompt, generationConfig, safetySettings);
            keyPoints = keyPointsResult.response.text().trim();
        }


        res.json({ summary, keyPoints });

    } catch (error) {
        console.error('Error summarizing text:', error);
        if (error.response && error.response.data) {
            console.error('Gemini API Error details:', error.response.data);
        }
        res.status(500).json({ error: 'Failed to summarize text. Please check your API key and input.', details: error.message });
    }
});

// 3. Image Analysis
app.post('/api/analyze-image', async (req, res) => {
    const { imageData } = req.body; // imageData is base64 string without data:image/jpeg;base64, prefix

    if (!imageData) {
        return res.status(400).json({ error: 'Missing image data in request.' });
    }

    try {
        const model = genAI.getGenerativeModel({ model: VISION_MODEL_NAME });

        // Helper to convert base64 to GoogleGenerativeAI format
        const imageToGenerativePart = (base64String, mimeType) => {
            return {
                inlineData: {
                    data: base64String,
                    mimeType // E.g., 'image/jpeg', 'image/png'
                },
            };
        };

        // Assuming common image types; you might need to detect mimeType on frontend
        // For simplicity, we'll assume jpeg or png based on common usage
        const mimeType = req.body.imageMimeType || 'image/jpeg'; // You might send this from frontend

        const imagePart = imageToGenerativePart(imageData, mimeType);

        const prompt = "Describe this image in detail, including objects, actions, and overall context. Keep the description concise, 2-3 sentences.";

        const result = await model.generateContent([prompt, imagePart], generationConfig, safetySettings);
        const response = await result.response;
        const description = response.text().trim();

        res.json({ description });

    } catch (error) {
        console.error('Error analyzing image:', error);
        if (error.response && error.response.data) {
            console.error('Gemini API Error details:', error.response.data);
        }
        res.status(500).json({ error: 'Failed to analyze image. Please ensure your API key is correct and the image is valid.', details: error.message });
    }
});

// 4. Voice/Audio Analysis (Placeholder - More Complex)
// Note: Direct audio transcription by Gemini models usually requires specialized
// audio processing on the backend (e.g., converting to FLAC, chunking)
// or integration with a dedicated Speech-to-Text API (like Google Cloud Speech-to-Text).
// This endpoint is a placeholder and will return a simulated response.
app.post('/api/analyze-audio', async (req, res) => {
    const { audioData } = req.body; // audioData would be a base64 string or similar

    if (!audioData) {
        // This is just a warning, as the client might call analyzeAudio without a file
        // if only the "start recording" button is pressed.
        console.warn("Received audio analysis request without audio data (likely from placeholder recording).");
    }

    try {
        // --- Placeholder for actual audio processing and AI call ---
        // In a real scenario:
        // 1. Save audioData to a temporary file.
        // 2. Use an audio processing library (e.g., fluent-ffmpeg) to convert/prepare for STT.
        // 3. Send to Google Cloud Speech-to-Text API or another specialized STT service.
        // 4. Then, send the *transcribed text* to Gemini for emotion/sentiment analysis.

        await new Promise(resolve => setTimeout(resolve, 2500)); // Simulate processing time

        const transcription = "This is a simulated transcription of your audio. For real transcription, consider using a dedicated Speech-to-Text API.";
        const emotion = "Simulated emotion: Neutral, with hints of curiosity. (Real emotion analysis would be derived from the transcribed text using a language model).";

        res.json({ transcription, emotion });

    } catch (error) {
        console.error('Error analyzing audio (simulated):', error);
        res.status(500).json({ error: 'Failed to analyze audio (simulated).', details: error.message });
    }
});

// 5. AI Resume Builder Endpoints
app.post('/api/resume/generate-summary', async (req, res) => {
    const { existingSummary, role, experienceLevel } = req.body;

    const prompt = `Generate a concise, impactful professional summary for a ${role} with ${experienceLevel} experience. Incorporate this existing information: "${existingSummary}". Focus on achievements and relevant skills. Keep it to 3-5 sentences.`;

    try {
        const model = genAI.getGenerativeModel({ model: TEXT_MODEL_NAME });
        const result = await model.generateContent(prompt, generationConfig, safetySettings);
        const response = await result.response;
        const generatedSummary = response.text().trim();
        res.json({ generatedSummary });
    } catch (error) {
        console.error('Error generating resume summary:', error);
        res.status(500).json({ error: 'Failed to generate summary. Please check your API key.', details: error.message });
    }
});

app.post('/api/resume/optimize-skills', async (req, res) => {
    const { currentSkills, jobDescriptionKeywords } = req.body;

    const prompt = `Optimize the following skills for a resume, considering these job requirements/keywords: "${jobDescriptionKeywords}". Suggest 5-10 strong, relevant skills based on "${currentSkills}". Provide as a comma-separated list of skills.`;

    try {
        const model = genAI.getGenerativeModel({ model: TEXT_MODEL_NAME });
        const result = await model.generateContent(prompt, generationConfig, safetySettings);
        const response = await result.response;
        const optimizedSkills = response.text().trim();
        res.json({ optimizedSkills });
    } catch (error) {
        console.error('Error optimizing skills:', error);
        res.status(500).json({ error: 'Failed to optimize skills. Please check your API key.', details: error.message });
    }
});


// Start the server
app.listen(port, () => {
    console.log(`Backend server listening at http://localhost:${port}`);
    console.log(`Open your frontend file (index.html) in your browser.`);
    console.log('Remember to set your GEMINI_API_KEY in your .env file!');
});