// ResultsScreen.js
import React from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import CsvTable from './CsvTable';
const HIDDEN_COURSES = [
  "Course",
  "PRACTICE SCHOOL I",
  "PRACTICE SCHOOL II",
  "THESIS"
];

const shouldShowCourse = (name) =>
  !HIDDEN_COURSES.includes(name.trim().toUpperCase());

export default function ResultsScreen({ route }) {
    const { results, preferredElective } = route.params;

  const statsMap = {};

  (results.gradeStats || [])
    .slice(1)
    .forEach(row => {
      const [courseCode, courseName, mean, std, count] = row;
      statsMap[courseName] = { mean, std, count };
    });
    

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>🏆 Results</Text>

      <Text style={styles.heading}>Recommended Courses Using Collaborative Filtering</Text>

{results.useCase1
  .filter(c => c[0] && c[0].trim().toUpperCase() !== "COURSE" && shouldShowCourse(c[0]))
  .map((c, i) => {

    return (
      <View key={i} style={{ marginBottom: 6 }}>
        <Text style={styles.text}>• {c[0]}</Text>

      </View>
    );
  })}
        <Text style={styles.heading}>Recommended Courses Using Robust FM</Text>
        {results.useCase2
  .filter(c => c[0] && c[0].trim().toUpperCase() !== "COURSE" && shouldShowCourse(c[0]))
  .map((c, i) => {

    return (
      <View key={i} style={{ marginBottom: 6 }}>
        <Text style={styles.text}>• {c[0]}</Text>

      </View>
    );
  })}
        <Text style={styles.heading}>Predicted Next Semester CGPA</Text>
        <Text style={styles.text}>{results.nextSemesterCGPA}</Text>

        <Text style={styles.heading}>Predicted Final CGPA</Text>
        <Text style={styles.text}>{results.predictedCGPA}</Text>

      <Text style={styles.heading}>Learning Pathways</Text>
      {results.learningPathways
        .filter(c => c[0] && c[0].trim().toUpperCase() !== "COURSE" && shouldShowCourse(c[0]))
        .map((c, i) => (
          <Text key={i} style={styles.text}>{c[0]}</Text>
        ))}

      <Text style={styles.heading}>Elective Personalization</Text>
      <Text style={styles.text}>Preferred: {preferredElective}</Text>
      {results.electivesPersonalization
        .filter(c => c[0] && c[0].trim().toUpperCase() !== "COURSE" && shouldShowCourse(c[0]))
        .map((c, i) => (
          <Text key={i} style={styles.text}>{c[0]}</Text>
        ))}
      <CsvTable csvText='output_3_stats.csv'/>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#fff'
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#5a189a'
  },
  heading: {
    fontSize: 18,
    marginTop: 16,
    fontWeight: '600',
    color: '#5a189a'
  },
  text: {
    fontSize: 15,
    marginVertical: 4,
    color: '#3d246c'
  }
});
