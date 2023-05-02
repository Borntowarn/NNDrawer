import React from 'react'
import './DropNode.css'

export default function DropNode({title}) {
  const handleClick = () => {
    console.log('clicked')
  }

  return (
    <div onClick={() => handleClick()} className='dropnode'>{title}</div>
  )
}
