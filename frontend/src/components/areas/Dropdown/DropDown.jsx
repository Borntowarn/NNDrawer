import React from 'react'
import DropNode from './DropNode/DropNode'
import './DropDown.css'

export default function DropDown({data, mode}) {
  return (
    <div className={mode ? 'dropdown' : 'dropdown active'}>
      {data.map((project, i) => (
        <DropNode title={project} key={i}/>
      ))}
    </div>
  )
}
