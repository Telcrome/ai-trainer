import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class DatasetService {

  constructor() { }

  getVersion(): String {
    return "1.00000 MOCK"
  }
}
