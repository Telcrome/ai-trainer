import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class LogServiceService {

  constructor(
    private http: HttpClient
  ) { }

  welcome = 'Test';

}
