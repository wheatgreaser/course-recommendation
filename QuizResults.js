import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import axios from 'axios';

async function sendToLLM(courseName, questions) {
  const response = await axios.post(
    'http://10.0.2.2:11434/api/generate',
    {
      model: "deepseek-r1:7b",
      prompt: `Recommend the user prerequisite university courses, course materials and textbooks considering that they got the following questions wrong: ${questions}

Return ONLY the courses they should take in a JSON array format like this:
{
  "topics": [
    "Topic 1",
    "Topic 2",
    "Topic 3"
  ]
}
  Return ONLY valid JSON. No explanation. No extra text.
If you cannot comply, return:
{ "topics": [] }`
,
      stream: false,
      format: "json"
    }
  );

 
  console.log("LLM response:", response.data);

  const parsed = safeParseJSON(response.data.response);

  if (!parsed || !parsed.topics) {
    throw new Error("Invalid LLM JSON");
  }

  return parsed;
}
function safeParseJSON(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    console.log("Raw LLM output:", str);
    return null;
  }
}
export default function QuizResults({ route }) {
  const { course, wrongQuestions } = route.params;


  const [recs, setRecs] = useState(null);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [score, setScore] = useState(null);

  useEffect(() => {
    async function fetchRecs() {
      try {
        const result = await sendToLLM(course, wrongQuestions);
        setRecs(result);
      } catch (err) {
        console.log("Parse error:", err);
      }
    }

    fetchRecs();
  }, [course, wrongQuestions]);


  if (!recs) {
    return (
      <View style={{ padding: 20 }}>
        <Text>Loading recommendations...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={{ padding: 20 }}>
      <Text style={{ fontSize: 22, fontWeight: 'bold', marginBottom: 20 }}>
        {course} learning recommendations
      </Text>
      {wrongQuestions.length < 3 && (
        <Text style={{ marginBottom: 15 }}>
          You can take the course {course} directly, but reviewing these topics will help you get the most out of it:
        </Text>
      )}
      {recs.topics.map((topic, index) => (
        <View key={index} style={{ marginBottom: 25 }}>
          <Text style={{ fontWeight: 'bold', marginBottom: 10 }}>
            {index + 1}. {topic}
          </Text>
        </View>
      ))}
    </ScrollView>
  );
}