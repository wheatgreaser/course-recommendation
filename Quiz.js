import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import axios from 'axios';
import { useNavigation } from '@react-navigation/native';

async function sendToLLM(courseName) {
  const response = await axios.post(
    'http://10.0.2.2:11434/api/generate',
    {
      model: "deepseek-r1:7b",
      prompt: `Generate a 10 question multiple choice quiz for checking if the user has the necessary prior knowledge for the topic: ${courseName}. NOTE: the questions should not be from the topic itself but rather from prerequisite topics. Each question should have 4 options with only one correct answer.

Return ONLY valid JSON in this format:

{
  "questions": [
    {
      "question": "string",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "A"
    }
  ]
}
`,
      stream: false,
      format: "json"
    }
  );
  console.log("LLM response:", response.data);

  const parsed = extractJSON(response.data.response);

  return parsed;
}

function extractJSON(raw) {
  try {
    return JSON.parse(raw);
  } catch {}

  try {
    const cleaned = raw.trim();

    const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error("No JSON found");

    const extracted = jsonMatch[0];

    return JSON.parse(extracted);
  } catch (e) {
    console.log("FAILED RAW:", raw);
    throw new Error("Invalid LLM JSON");
  }
}

export default function Quiz({ route }) {
  const navigation = useNavigation();
  const { course } = route.params;
  const [ questions, setQuestions ] = useState();
  const [quiz, setQuiz] = useState(null);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [score, setScore] = useState(null);

  useEffect(() => {
    async function fetchQuiz() {
      try {
        const result = await sendToLLM(course);
        setQuiz(result);
      } catch (err) {
        console.log("Parse error:", err);
      }
    }

    fetchQuiz();
  }, [course]);

  const selectOption = (questionIndex, optionLetter) => {
    setSelectedAnswers(prev => ({
      ...prev,
      [questionIndex]: optionLetter
    }));
  };

  const calculateScore = () => {
    let correct = 0;

    quiz.questions.forEach((q, index) => {
      if (selectedAnswers[index] === q.answer) {
        correct++;
      }
    });

    setScore(correct);
  };
    const wrongQuestions = () => {
    
      const wrongQuestions = [];
    quiz.questions.forEach((q, index) => {
      if (selectedAnswers[index] !== q.answer) {
        wrongQuestions.push(q.question);
      }
    });

    setQuestions(wrongQuestions);
  };

  if (!quiz || !quiz.questions) {
  return (
    <View style={{ padding: 20 }}>
      <Text>Loading quiz...</Text>
    </View>
  );
}

  return (
    <ScrollView style={{ padding: 20 }}>
      <Text style={{ fontSize: 22, fontWeight: 'bold', marginBottom: 20 }}>
        {course} Eligibility Quiz
      </Text>

      {quiz.questions.map((q, index) => (
        <View key={index} style={{ marginBottom: 25 }}>
          <Text style={{ fontWeight: 'bold', marginBottom: 10 }}>
            {index + 1}. {q.question}
          </Text>

          {q.options.map((opt, i) => {
            const letter = String.fromCharCode(65 + i);
            const isSelected = selectedAnswers[index] === letter;

            return (
              <TouchableOpacity
                key={i}
                onPress={() => selectOption(index, letter)}
                style={{
                  padding: 10,
                  marginVertical: 5,
                  borderRadius: 8,
                  backgroundColor: isSelected ? '#4CAF50' : '#e0e0e0'
                }}
              >
                <Text>
                  {letter}. {opt}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      ))}

      <TouchableOpacity
        onPress={() => {
          let wrong = [];
          let correct = 0;

          quiz.questions.forEach((q, index) => {
            if (selectedAnswers[index] === q.answer) {
              correct++;
            } else {
              wrong.push(q.question);
            }
          });

          navigation.navigate("QuizResults", {
            course,
            score: correct,
            wrongQuestions: wrong
          });
        }}
        style={{
          backgroundColor: '#2196F3',
          padding: 15,
          borderRadius: 10,
          alignItems: 'center',
          marginBottom: 30
        }}
      >
        <Text style={{ color: 'white', fontWeight: 'bold' }}>
          Submit Quiz
        </Text>
  </TouchableOpacity>
    </ScrollView>
  );
}