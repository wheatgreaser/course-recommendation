import React, { useState,useEffect } from 'react';

export function useCourse() {
  const [course, setCourse ] = useState()


  return [course, setCourse]
}