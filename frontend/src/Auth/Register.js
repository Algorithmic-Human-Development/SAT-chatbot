import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { Input, message } from "antd";

const Register = ({ onRegister }) => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });
  const navigate = useNavigate();
  const [messageApi, contextHolder] = message.useMessage();
  const baseURL = process.env.REACT_APP_BASE_URL;

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        `${baseURL}/api/register/`,
        formData
      );

      // Store the user group from registration response if provided
      if (response.data.group) {
        localStorage.setItem("userGroup", response.data.group);
      }

      messageApi.open({
        type: "success",
        content: `ثبت‌نام کاربر موفقیت‌آمیز بود. لطفا وارد شوید.`,
      });

      // Redirect to login page after successful registration
      navigate("/login");
    } catch (error) {
      console.log("reg err: ", error);
      messageApi.open({
        type: "error",
        content: `ثبت نام کاربر با خطا مواجه شد`,
      });
    }
  };

  return (
    <>
      {contextHolder}
      <div className="flex items-center text-right justify-center min-h-screen bg-gray-100">
        <form
          onSubmit={handleSubmit}
          className="w-full  max-w-md bg-white shadow-lg rounded-lg p-6"
        >
          <h2 className="text-2xl font-bold mb-4">ثبت‌ نام</h2>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">نام‌ کاربری</label>
            <Input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:outline-none focus:ring focus:ring-blue-300"
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">ایمیل</label>
            <Input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:outline-none focus:ring focus:ring-blue-300"
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">رمزعبور</label>
            <Input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:outline-none focus:ring focus:ring-blue-300"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600"
          >
            ثبت‌ نام
          </button>
          <div className="text-center mt-3" dir="rtl">
            قبلا حساب ساخته‌اید؟
            <a
              className="text-blue-500 hover:text-blue-700 transition duration-200 mr-1"
              href="/login"
            >
              ورود!
            </a>
          </div>
        </form>
      </div>
    </>
  );
};

export default Register;
