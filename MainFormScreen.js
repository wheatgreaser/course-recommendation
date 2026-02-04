import React, { useState,useEffect } from 'react';
import { StyleSheet, Text, View, TextInput, ScrollView, Alert, ActivityIndicator, Image, TouchableOpacity } from 'react-native';
import { KeyboardAvoidingView, Platform } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import Button from './components/Button';
import theme from './theme';
import i18n from './i18n/en.json';
import { StatusBar } from 'expo-status-bar';
import { useNavigation } from '@react-navigation/native';
import Papa from "papaparse";



import axios from 'axios';
const subjects = [
  "ENGINEERING GRAPHICS",
  "COMPUTER PROGRAMMING",
  "GENERAL BIOLOGY",
  "BIOLOGY LABORATORY",
  "TECHNICAL REPORT WRITING",
  "THERMODYNAMICS",
  "WORKSHOP PRACTICE",
  "CHEMISTRY LABORATORY",
  "PHYSICS LABORATORY",
  "GENERAL CHEMISTRY",
  "MATHEMATICS I",
  "PROBABILITY & STATISTICS",
  "ENVIRONMENTAL STUDIES",
  "PRACTICE SCHOOL I",
  "MECH OSCILLATIONS & WAVE",
  "ELECTRICAL SCIENCES",
  "MATHEMATICS II",
  "MATHEMATICS III",
  "PRACTICE SCHOOL II",
  "PRINCIPLES OF ECONOMICS",
  "DIGITAL DESIGN",
  "MICROPROC & INTERFACING",
  "OBJECT ORIENTED PROG",
  "OPERATING SYSTEMS",
  "FUNDA OF FIN AND ACCOUNT",
  "CONTROL SYSTEMS",
  "ELECTRONIC DEVICES",
  "SIGNALS & SYSTEMS",
  "ANALOG ELECTRONICS",
  "MICROELECTRONIC CIRCUITS",
  "ELECTRICAL MACHINES",
  "STUDY PROJECT",
  "DERIVATIVES & RISK MGMT",
  "SECUR ANAL & PORTFOL MGT",
  "FLUID MECHANICS",
  "DESIGN PROJECT",
  "DATA STRUCTURES & ALGO",
  "FINANCIAL MANAGEMENT",
  "BUSS ANAL & VALUATION",
  "COMPUTER ARCHITECTURE",
  "COMMUNICATION SYSTEMS",
  "CRYPTOGRAPHY",
  "LOGIC IN COMPUTER SC",
  "LABORATORY PROJECT",
  "MECHANICS OF SOLIDS",
  "INTERNET OF THINGS",
  "PRINCIPLES OF MANAGEMENT",
  "HEAT TRANSFER",
  "DISASTER AND DEVELOPMENT",
  "DATABASE SYSTEMS",
  "ANALOG & DIGIT VLSI DES",
  "PRINCIPLES OF PROGG LANG",
  "COMPILER CONSTRUCTION",
  "COMPUTER NETWORKS",
  "DESIGN & ANAL OF ALGO",
  "THEORY OF COMPUTATION",
  "MACHINE LEARNING",
  "DISCR STRUC FOR COMP SCI",
  "OPTIMIZATION",
  "ELEC & ELECTRONIC CIRCUITS LAB",
  "POWER ELECTRONICS",
  "CONTROL SYSTEMS LABORATORY",
  "PROJECT APPRAISAL",
  "GAME THEORY AND ITS APPLICATIO",
  "ARTIFICIAL INTELLIGENCE",
  "ELECTROMAGNETIC THEO",
  "APPLIED THERMODYNAMICS",
  "ENGINEERING OPTIMIZATION",
  "NEURAL NET & FUZZY LOGIC",
  "PRIMEMOVERS & FLUID MACH",
  "DIGITAL IMAGE PROCESSING",
  "DIGITAL SIGNAL PROCESS",
  "DISCRETE MATHEMATICS",
  "COMMUNICATION NETWORKS",
  "EM FIELDS & MICRO ENGG",
  "SRIMAD BHAGAVAD GITA",
  "INFO THEORY & CODING",
  "THESIS",
  "HUMAN RESOURCE DEVELOP",
  "ELECTROMAGNETIC THEORY",
  "FINANCIAL RISK ANALYTICS & M",
  "FOUNDATIONS OF DATA SCIENCE",
  "CULTURAL STUDIES",
  "URBAN POLICY AND GOVERNANCE",
  "POWER SYSTEMS",
  "INFORMATION RETRIEVAL",
  "IC ENGINES",
  "MECHANICAL VIBRATIONS",
  "INTRODUCTION TO CRITICAL PEDAG",
  "COMPUTER AIDED DESIGN",
  "ADV MECHANICS OF SOLIDS",
  "KIN & DYN OF MACHINES",
  "PRODUCTION TECHNIQUES I",
  "MACHINE DESIGN & DRAWING",
  "PRODUCTION TECHNIQUES II",
  "MATERIALS SCIENCE & ENGG",
  "MECHANICAL ENGG LAB",
  "HUM THEO OF SC & TECH",
  "DATA MINING",
  "ECONOMETRIC METHODS",
  "INSTRU METHODS OF ANAL",
  "NUMBER THEORY",
  "LINGUISTICS",
  "COMPARATIVE INDIAN LIT",
  "SCIENCE,TECH & MODERNITY",
  "SUPPLY CHAIN MANAGEMENT",
  "MATHEMATIC & STAT METHOD",
  "PUBLIC ADMINISTRATION",
  "SOFTWARE ENGINEERING",
  "SEL TOPICS FROM COMP SC",
  "INTRODUCTORY PHILOSOPHY",
  "HUMAN COMP INTERACTION",
  "POP LITER & CULT S ASIA",
  "ELECTROMAGNETIC THEO I",
  "ADV COMMUNICATIVE ENG",
  "CREATIVE WRITING",
  "MONEY BANK & FIN MARKETS",
  "MICROECONOMICS",
  "BUSINESS COMMUNICATION",
  "MACROECONOMICS",
  "CONTEMPORARY INDIA",
  "BIOLOGICAL CHEMISTRY",
  "MICROBIOLOGY",
  "ECONOMIC ENV OF BUSINESS",
  "CONSTRUCTION PLAN & TECH",
  "SURVEYING",
  "HYDRAULIC ENGINEERING",
  "DES OF STEEL STRUCTURES",
  "DESIGN OF REINFORCED CONCRETE",
  "WATER & WASTEWATER TREAT",
  "HIGHWAY ENGINEERING",
  "SOIL MECHANICS",
  "ANALYSIS OF STRUCTURES",
  "ENGINEERING HYDROLOGY",
  "FOUNDATION ENGINEERING",
  "CIVIL ENGINEERING MATERIALS",
  "INTERNATIONAL ECONOMICS",
  "ECONOMIC ANAL OF PUB POL",
  "PUBLIC FIN THEO & POLICY",
  "APPLIED ECONOMETRICS",
  "ECONOMIC OF GROWTH & DEV",
  "POST COLONIAL LITERATURE",
  "ISSUES IN ECONOMIC DEV",
  "PHILOSOPHY OF NAGARJUNA",
  "PUBLIC POLICY",
  "APPLIED STATISTICAL METHODS",
  "ORDINARY DIFF EQUATIONS",
  "MATHEMATICAL METHODS",
  "INTRO TO GLOBALIZATION",
  "LOCAL GOVERNANCE AND PARTICIPA",
  "INTRO TO GENDER STUDIES",
  "GRAPHS AND NETWORKS",
  "ALGEBRA I",
  "CINEMATIC ADAPTATION",
  "ELEMENTARY REAL ANALYSIS",
  "NUMERICAL ANALYSIS",
  "PARTIAL DIFF EQUATIONS",
  "MEASURE & INTEGRATION",
  "INTRO TO FUNCTIONAL ANAL",
  "INTRODUCTION TO TOPOLOGY",
  "NATURAL LANGUAGE PROCESSING",
  "OPERATIONS RESEARCH",
  "DIFFERENTIAL GEOMETRY",
  "INTRODUCTION TO PHONOLOGY",
  "NEGOTIATION SKILLS AND TECHNIQ",
  "CHEMICAL ENGG LAB II",
  "CONTEMPORARY INDAIN ENG FIC",
  "INTERNATIONAL BUSINESS",
  "PROCESS DES PRINCIPLE II",
  "PROCESS DES PRINCIPLES I",
  "PROCESS DYN & CONTROL",
  "SEPARATION PROCESSES I",
  "SEPARATION PROCESSES II",
  "CHEMICAL PROCESS CALCULA",
  "INDUS INSTRUMENT & CONT",
  "CHEMICAL ENGG LAB I",
  "TRANSD & MEASUREMENT TEC",
  "NUM METHOD FOR CHEM ENGG",
  "MATERIAL SCIENCE & ENGG",
  "CHEM ENGG THERMODYNAMICS",
  "KINETICS & REACTOR DESIG",
  "ELECTRO INST & INST TECH",
  "INTRODUCTION TO MEMS",
  "ARTIFICIAL INTELLIGENCE FOR ROBOTS",
  "SUSTAINABLE MANUFACTURING",
  "EFFECTIVE PUBLIC SPEAKING",
  "ENGINEERING CHEMISTRY",
  "COMBINATORIAL MATHEMATICS",
  "AUTOMOTIVE TECHNOLOGY",
  "SCIENCE OF SUSTAINABLE HAPPINESS",
  "LITERARY CRITICISM",
  "DYNAMICS OF SOCIAL CHANGE",
  "ROBOTICS",
  "PHONETICS & SPOKEN ENGLISH",
  "COMPUTER GRAPHICS",
  "COMPUTATIONAL PHYSICS",
  "QUANTUM MECHANICS I",
  "NONLINEAR OPTIMIZATION",
  "CLASSICAL MECHANICS",
  "ENVIRONMENTAL POLLUTION CONTROL",
  "MATERIALS SCIENCE AND ENGINEERING",
  "WIND ENERGY",
  "DEVELOPMENT ECONOMICS",
  "STATISTICAL MECHANICS",
  "ENGINES, MOTORS, AND MOBILITY",
  "COMPUTER-AIDED DESIGN",
  "MANUFACTURING PROCESSES",
  "MANUFACTURING MANAGEMENT",
  "MECHANISMS AND MACHINES",
  "DESIGN OF MACHINE ELEMENTS",
  "ADVANCED PHYSICS LAB",
  "ADVANCED MECHANICS OF SOLIDS",
  "MATHEMATICAL METHODS OF PHYSICS",
  "VIBRATIONS AND CONTROL",
  "ELECTROMAGNETIC THEORY AND APPLICATIONS",
  "MODERN POLITICAL CONCEPTS",
  "ADVANCED MANUFACTURING PROCESSES",
  "ELECTROMAGNETIC THEORY II",
  "AIRPORT, RAIL & WATERWAYS ENGINEERING",
  "SOLID STATE PHYSICS",
  "STRUCTURAL DYNAMICS",
  "ENVIRONMENTAL DEVELOPMENT & CLIMATE CHANGE",
  "OPTICS",
  "MODERN PHYSICS LAB",
  "ATOMIC & MOLECULAR PHYSICS",
  "NUCLEAR & PARTICLE PHYSICS",
  "COMPUTATIONAL GEOMETRY",
  "CRITICAL ANALYSIS OF LITERATURE & CINEMA",
  "QUANTUM MECHANICS II",
  "INTRODUCTION TO MASS COMMUNICATION",
  "ENERGY MANAGEMENT",
  "MECHATRONICS & AUTOMATION",
  "FUEL CELL SCIENCE AND TECHNOLOGY",
  "INTRODUCTION TO MOLECULAR BIOLOGY",
  "ANATOMY, PHYSIOLOGY & HYGIENE",
  "BIOETHICS & BIOSAFETY",
  "DISPENSING PHARMACY",
  "PHARMACEUTICAL FORMULATION & BIOPHARMACEUTICS",
  "FORENSIC PHARMACY",
  "PROCESS ENGINEERING",
  "THIN FILM TECHNOLOGY",
  "PHYSICAL PHARMACY",
  "INTRODUCTION TO NANOSCIENCE",
  "MEDICINAL CHEMISTRY I",
  "MEDICINAL CHEMISTRY II",
  "PHARMACOLOGY II",
  "PHARMACOLOGY I",
  "PHARMACEUTICAL CHEMISTRY",
  "PHARMACEUTICAL ANALYSIS",
  "NATURAL DRUGS",
  "BEHAVIORAL ECONOMICS",
  "STATISTICAL INFERENCE & APPLICATIONS",
  "COLLOIDS AND INTERFACE ENGINEERING",
  "PHYSICAL CHEMISTRY II",
  "ORGANIC CHEMISTRY I",
  "FPGA BASED SYSTEM DESIGN LAB"
];
const axiosInstance = axios.create({
  timeout: 10000, // 10 seconds timeout
});
const SERVER_URL = "http://10.0.2.2:10000" ;
const HIDDEN_COURSES = [
  "Course",
  "PRACTICE SCHOOL I",
  "PRACTICE SCHOOL II",
  "THESIS"
];

const shouldShowCourse = (name) =>
  !HIDDEN_COURSES.includes(name.trim().toUpperCase());
export default function MainFormScreen() {
  const navigation = useNavigation();
  const [error, setError] = useState("");
  const [singleDegree, setSingleDegree] = useState('None');
  const [dualDegree, setDualDegree] = useState('None');
  const [semester, setSemester] = useState('1-1');
  const [subject, setSubject] = useState('');
  const [courseGrade, setGrade] = useState('10');
  const [preferredElective, setPreferredElective] = useState('');
  const [allCourses, setAllCourses] = useState({});
  const [filteredSubjects, setFilteredSubjects] = useState(
    subjects.filter(shouldShowCourse)
  );
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [csv1Variable, setCsv1Variable] = useState([]);
  const [csv2Variable, setCsv2Variable] = useState([]);
  const [csv3Variable, setCsv3Variable] = useState([]);
  const [csv4Variable, setCsv4Variable] = useState([]);
  const [csv5Variable, setCsv5Variable] = useState([]);
  const [csv7Variable, setCsv7Variable] = useState([]);

  
  
  const addCourse = () => {
  if (subject.trim() === '') {
    Alert.alert("Input Error", "Subject field cannot be empty.");
    return;
  }
  const semesterCourses = allCourses[semester] || [];


  const newCourse = { subject, courseGrade };

  setAllCourses(prevCourses => {
    const updatedCourses = { ...prevCourses };
    if (!updatedCourses[semester]) updatedCourses[semester] = [];

    // Check if course already exists in that semester
    if (updatedCourses[semester].some(c => c.subject === subject)) {
      Alert.alert("Duplicate Course", "This course is already added for the selected semester.");
      return prevCourses;
    }

    updatedCourses[semester].push(newCourse);
    return updatedCourses;
  });

  setSubject('');
  setGrade('10');
  setFilteredSubjects(subjects);
  setSearchTerm('');
};


  
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const waitForResults = async (retries = 8, delay = 4000) => {
  for (let i = 0; i < retries; i++) {
    try {
      const csv7 = await fetchCSVData('csv7');
      if (csv7 && csv7.length > 1) {
        await fetchAllCSVData();
        return;
      }
    } catch (e) {}
    await new Promise(res => setTimeout(res, delay));
  }
  Alert.alert("Timeout", "Results took too long to generate.");
};
  const submitForm = async () => {
    const formData = {
      ...allCourses,
      beDegree: singleDegree,
      mscDegree: dualDegree,
      preferredElective: preferredElective
    };

    setIsProcessing(true);
    setIsSubmitted(false);
    Alert.alert('Submitting', 'Submitting your form...');
    try {
      const response = await axios.post(`${SERVER_URL}/submit-form`, formData,{
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setIsSubmitted(true);
      Alert.alert('Submitted', 'Form submitted successfully! Waiting for results...');

      waitForResults();
    } catch (error) {
      let msg = 'Unknown error';
      if (error.response) {
        msg = `Server error: ${error.response.status}`;
      } else if (error.request) {
        msg = 'No response from server. Please check your connection.';
      } else {
        msg = `Request failed: ${error.message}`;
      }
      if (error.code === 'ECONNABORTED') {
        msg = 'Request timed out. Please try again.';
      }
      setError(msg);
      Alert.alert('Error', msg);
    } finally {
      setIsProcessing(false);
    }
  };

const fetchAllCSVData = async () => {
  try {
    const [csv1Data, csv2Data, csv3Data, csv4Data, csv5Data, csv7Data] = await Promise.all([
      fetchCSVData('csv1'),
      fetchCSVData('csv2'),
      fetchCSVData('csv3'),
      fetchCSVData('csv4'),
      fetchCSVData('csv5'),
      fetchCSVData('csv7'),


    ]);

    const computedResults = {
      useCase1: csv1Data,
      predictedCGPA: csv2Data,
      learningPathways: csv3Data,
      electivesPersonalization: csv4Data,
      nextSemesterCGPA: csv5Data,
      useCase2: csv7Data,

    };

    navigation.navigate("Results", {
      results: computedResults,
      preferredElective
    });

  } catch (error) {
    console.error(error);
    Alert.alert("Error", "Failed to fetch results");
  }
};




const fetchCSVData = async (csvId) => {
  try {
    const response = await axios.get(`${SERVER_URL}/get-csv-data/${csvId}`);

    // Guard: no data
    if (!response.data) {
      return (csvId === 'csv2' || csvId === 'csv5') ? "" : [];
    }

    // Scalar CSVs (single value)
    if (csvId === 'csv2' || csvId === 'csv5') {
      return response.data.toString().trim();
    }

    // Multi-row CSVs
    return response.data
      .toString()
      .split('\n')
      .map(row => row.split(','));

  } catch (error) {
    console.error(`Error fetching ${csvId} data:`, error);
    return (csvId === 'csv2' || csvId === 'csv5') ? "" : [];
  }
};

  
const displayResults = (csv1Data, csv2Data, csv3Data, csv4Data, csv5Data, csv7Data) => {
  setResults({
    useCase1: Array.isArray(csv1Data) && csv1Data.length > 0 
      ? csv1Data 
      : ["No data available"],

    predictedCGPA: csv2Data !== '' 
      ? csv2Data 
      : "Data not available",

    learningPathways: Array.isArray(csv3Data) && csv3Data.length > 0 
      ? csv3Data 
      : ["No data available"],

    electivesPersonalization: Array.isArray(csv4Data) && csv4Data.length > 0 
      ? csv4Data 
      : ["No data available"],

    nextSemesterCGPA: csv5Data !== '' && csv5Data !== null
      ? csv5Data
      : "Data not available",

    useCase2: Array.isArray(csv7Data) && csv7Data.length > 0 
      ? csv7Data 
      : ["No data available"],

  });
};

  
  const handleSubjectChange = (text) => {
    setSearchTerm(text);
    setSubject(text);
    setFilteredSubjects(subjects.filter(sub => sub.toLowerCase().includes(text.toLowerCase())));
  };


  const currentCourses = allCourses[semester] || [];
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const [isSubjectPickerVisible, setIsSubjectPickerVisible] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  
  return (
    <KeyboardAvoidingView behavior="padding" style={{ flex: 1 }}>
      <View style={styles.gradientBg}>
        <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
          <StatusBar style="light" />
          {/* Hero Header with Logo */}
          <View style={styles.heroHeader}>
            <Image source={{uri: 'https://img.icons8.com/color/96/000000/graduation-cap.png'}} style={styles.logo} />
            <Text style={styles.heroTitle}>Course Recommendation</Text>
            <Text style={styles.heroSubtitle}>Personalized academic planning made easy</Text>
          </View>

        {/* Degree & Semester Section */}
        <View style={styles.sectionCardFloating}>
          <Text style={styles.sectionTitle}><Text style={styles.sectionIcon}>🎓</Text> Your Program</Text>
          <Text style={styles.label}>Single Degree</Text>
          <Picker selectedValue={singleDegree} style={styles.picker} onValueChange={setSingleDegree}>
            <Picker.Item label="None" value="None" />
            <Picker.Item label="B.E Chemical" value="B.E Chemical" />
            <Picker.Item label="B.E Civil" value="B.E Civil" />
            <Picker.Item label="B.E Computer Science" value="B.E Computer Science" />
            <Picker.Item label="B.E Electrical & Electronic" value="B.E Electrical & Electronic" />
            <Picker.Item label="B.E Electronics & Communication" value="B.E Electronics & Communication" />
            <Picker.Item label="B.E Electronics and Instrumentation" value="B.E Electronics and Instrumentation" />
            <Picker.Item label="B.E Mechanical" value="B.E Mechanical" />
            <Picker.Item label="B.Pharm" value="B.Pharm" />
          </Picker>
          <Text style={styles.label}>Dual Degree</Text>
          <Picker selectedValue={dualDegree} style={styles.picker} onValueChange={setDualDegree}>
            <Picker.Item label="None" value="None" />
            <Picker.Item label="M.Sc. Economics" value="M.Sc. Economics" />
            <Picker.Item label="M.Sc. Bio" value="M.Sc. Bio" />
            <Picker.Item label="M.Sc. Physics" value="M.Sc. Physics" />
            <Picker.Item label="M.Sc. Chemistry" value="M.Sc. Chemistry" />
            <Picker.Item label="M.Sc. Mathematics" value="M.Sc. Mathematics" />
          </Picker>
          <Text style={styles.label}>Semester</Text>
          <Picker selectedValue={semester} style={styles.picker} onValueChange={setSemester}>
            <Picker.Item label="1-1" value="1-1" />
            <Picker.Item label="1-2" value="1-2" />
            <Picker.Item label="2-1" value="2-1" />
            <Picker.Item label="2-2" value="2-2" />
            <Picker.Item label="3-1" value="3-1" />
            <Picker.Item label="3-2" value="3-2" />
            <Picker.Item label="4-1" value="4-1" />
            <Picker.Item label="4-2" value="4-2" />
          </Picker>
        </View>

        {/* Courses Section */}
        <View style={styles.sectionCardFloating}>
          <Text style={styles.sectionTitle}><Text style={styles.sectionIcon}>📚</Text> Your Courses</Text>
          <Text style={styles.label}>Selected Courses</Text>
          {currentCourses.length === 0 ? (
            <Text style={styles.emptyText}>No courses added yet.</Text>
          ) : (
            currentCourses.map((item, idx) => (
              <TouchableOpacity key={idx} style={styles.courseItem} onPress={() => Alert.alert(
                "Delete Course",
                "Do you want to delete this entry?",
                [
                  { text: "No", style: "cancel" },
                  { text: "Yes", onPress: () => {
                    setAllCourses((prevCourses) => {
                      const updatedCourses = { ...prevCourses };
                      updatedCourses[semester] = updatedCourses[semester].filter((_, i) => i !== idx);
                      return updatedCourses;
                    });
                  }}
                ]
              )}>
                <Text style={styles.courseText}>{item.subject}: {item.courseGrade}</Text>
              </TouchableOpacity>
            ))
          )}
        </View>

        {/* Add Course Section */}
        <View style={styles.sectionCardFloating}>
          <Text style={styles.sectionTitle}><Text style={styles.sectionIcon}>➕</Text> Add a Course</Text>
          <Text style={styles.label}>Subject</Text>
          <TextInput
            style={styles.input}
            placeholder="Search or select a subject"
            value={searchTerm}
            onChangeText={(text) => { setSearchTerm(text); setDropdownVisible(true); }}
            onFocus={() => setDropdownVisible(true)}
          />
          {dropdownVisible && (
            <ScrollView style={styles.suggestionList}>
              {filteredSubjects
                .filter((item) => item.toLowerCase().includes(searchTerm.toLowerCase()))
                .map((item, index) => (
                  <TouchableOpacity key={index} onPress={() => { setSubject(item); setDropdownVisible(false); setSearchTerm(item); }}>
                    <Text style={styles.suggestionItem}>{item}</Text>
                  </TouchableOpacity>
                ))}
            </ScrollView>
          )}
          <Text style={styles.label}>Grade</Text>
          <Picker selectedValue={courseGrade} style={styles.picker} onValueChange={setGrade}>
            <Picker.Item label="A" value="10" />
            <Picker.Item label="A-" value="9" />
            <Picker.Item label="B" value="8" />
            <Picker.Item label="B-" value="7" />
            <Picker.Item label="C" value="6" />
            <Picker.Item label="C-" value="5" />
            <Picker.Item label="D" value="4" />
            <Picker.Item label="E" value="2" />
          </Picker>
          <Text style={styles.note}>Note: If you have a course with a non-numeric grade, do not enter it.</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={() => { addCourse(); setDropdownVisible(false); setSearchTerm(""); }}>
            <Text style={styles.primaryButtonText}>Add Course</Text>
          </TouchableOpacity>
        </View>

        {/* Elective Section */}
        <View style={styles.sectionCardFloating}>
          <Text style={styles.sectionTitle}><Text style={styles.sectionIcon}>⭐</Text> Preferred Elective</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter your preferred elective"
            value={preferredElective}
            onChangeText={(text) => setPreferredElective(text)}
          />
        </View>

        {/* Submit Section */}
        <View style={styles.sectionCardFloating}>
          <TouchableOpacity style={styles.primaryButton} onPress={submitForm} disabled={isProcessing}>
            <Text style={styles.primaryButtonText}>{isProcessing ? "Processing..." : "Submit"}</Text>
          </TouchableOpacity>
          {isProcessing && <ActivityIndicator size="large" color="#5a189a" style={{ marginTop: 12 }} />}
          {isSubmitted && !results && (
            <Text style={styles.processingText}>Processing your request. Results will be available shortly.</Text>
          )}
          {error ? (
            <Text style={styles.errorText}>{error}</Text>
          ) : null}
        </View>
          {results && (
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() =>
            navigation.navigate("Results", {
              results,
              preferredElective,
            })
          }
        >
          <Text style={styles.primaryButtonText}>View Results</Text>
        </TouchableOpacity>
  )}

        
        </ScrollView>
      </View>
    </KeyboardAvoidingView>
  );
  
  

  
}

const styles = StyleSheet.create({
  gradientBg: {
    flex: 1,
    backgroundColor: 'linear-gradient(180deg, #a18cd1 0%, #fbc2eb 100%)', // fallback for web, not native
    backgroundColor: '#a18cd1', // fallback for native
  },
  scrollContainer: {
    padding: 0,
    paddingBottom: 32,
    backgroundColor: 'transparent',
  },
  heroHeader: {
    backgroundColor: 'rgba(90,24,154,0.95)',
    paddingTop: 48,
    paddingBottom: 32,
    paddingHorizontal: 24,
    borderBottomLeftRadius: 32,
    borderBottomRightRadius: 32,
    alignItems: 'center',
    marginBottom: 18,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
  },
  logo: {
    width: 64,
    height: 64,
    marginBottom: 8,
  },
  heroTitle: {
    color: '#fff',
    fontSize: 34,
    fontWeight: 'bold',
    letterSpacing: 1.5,
    marginBottom: 6,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Bold' : 'Roboto',
  },
  heroSubtitle: {
    color: '#e0c3fc',
    fontSize: 16,
    fontWeight: '400',
    letterSpacing: 0.5,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Regular' : 'Roboto',
  },
  sectionCardFloating: {
    backgroundColor: '#fff',
    marginHorizontal: 18,
    marginBottom: 24,
    borderRadius: 22,
    padding: 20,
    shadowColor: '#a18cd1',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 18,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    elevation: 8,
  },
  sectionTitle: {
    fontSize: 21,
    fontWeight: '700',
    color: '#5a189a',
    marginBottom: 12,
    letterSpacing: 0.7,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Bold' : 'Roboto',
  },
  sectionIcon: {
    fontSize: 20,
    marginRight: 4,
  },
  label: {
    fontSize: 16,
    marginBottom: 6,
    fontWeight: '600',
    color: '#5a189a',
    letterSpacing: 0.2,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Medium' : 'Roboto',
  },
  picker: {
    height: 48,
    width: '100%',
    backgroundColor: '#f3f0fa',
    borderRadius: 10,
    marginBottom: 12,
    marginTop: 2,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Regular' : 'Roboto',
  },
  input: {
    height: 46,
    borderColor: '#b39ddb',
    borderWidth: 1.5,
    marginBottom: 14,
    paddingHorizontal: 16,
    borderRadius: 10,
    backgroundColor: '#f6f7fb',
    fontSize: 16,
    color: '#3d246c',
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Regular' : 'Roboto',
  },
  primaryButton: {
    backgroundColor: '#a18cd1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 2,
    shadowColor: '#a18cd1',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.18,
    shadowRadius: 8,
    elevation: 4,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 0.7,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Bold' : 'Roboto',
  },
  courseItem: {
    padding: 14,
    backgroundColor: '#f3f0fa',
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    elevation: 2,
  },
  courseText: {
    fontSize: 16,
    color: '#3d246c',
    fontWeight: '500',
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Medium' : 'Roboto',
  },
  suggestionList: {
    maxHeight: 120,
    marginTop: 5,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 10,
    backgroundColor: '#fff',
    shadowColor: '#a18cd1',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  suggestionItem: {
    padding: 14,
    fontSize: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f0fa',
    color: '#3d246c',
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Regular' : 'Roboto',
  },
  note: {
    color: '#7c6f98',
    marginTop: 7,
    fontSize: 13,
    fontStyle: 'italic',
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Italic' : 'Roboto',
  },
  emptyText: {
    color: '#b39ddb',
    fontSize: 15,
    fontStyle: 'italic',
    marginBottom: 8,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Regular' : 'Roboto',
  },
  processingText: {
    marginTop: 20,
    fontSize: 16,
    textAlign: 'center',
    color: '#5a189a',
    fontWeight: '500',
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Medium' : 'Roboto',
  },
  errorText: {
    color: '#d7263d',
    textAlign: 'center',
    marginVertical: 10,
    fontWeight: '600',
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Bold' : 'Roboto',
  },
  subHeading: {
    fontSize: 17,
    fontWeight: '600',
    marginTop: 14,
    marginBottom: 6,
    color: '#5a189a',
    letterSpacing: 0.3,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Medium' : 'Roboto',
  },
  resultText: {
    fontSize: 15,
    color: '#3d246c',
    marginBottom: 2,
    fontFamily: Platform.OS === 'ios' ? 'AvenirNext-Regular' : 'Roboto',
  },
});