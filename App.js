// App.js (root)
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import MainFormScreen from './MainFormScreen';
import Results from './Results';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen 
          name="Form"
          component={MainFormScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Results"
          component={Results}
          options={{ title: "Your Results" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
