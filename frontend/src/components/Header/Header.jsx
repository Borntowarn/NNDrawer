import './Header.css'

export default function Header() {
  const handleClick = () => {
    
  } 

  return (
    <div className='header-area'>
        <div className='header-title'>
            NNdrawer
        </div>
        <a onClick={() => handleClick()} className='user-logo'>
            <img src="images/user.png" />
        </a>
    </div>
  )
}
